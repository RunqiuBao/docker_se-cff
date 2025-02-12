import torch.nn as nn
import torch
from torch import Tensor
import torchvision
from typing import List, Dict, Tuple
from thop import profile
import copy
import time
import torch.nn.functional as F

from ..rtdetr.rtdetr import RTDETR
from ..concentration import ConcentrationNet
from ..rtdetr.rtdetr_criterion import RTDETRCriterion
from ..utils.misc import freeze_module_grads
from ..rtdetr.matcher import HungarianMatcher
from mmdet.registry import MODELS
from mmdet.structures.mask import mask_target, BitmapMasks
from mmengine.config import Config

from ...methods.visz_utils import RenderImageWithBboxes


class EventPickTargetPredictionNetwork(nn.Module):
    _skip_pickable_region = None
    _skip_rtdetr = None

    def __init__(
        self,
        concentration_net_cfg: dict,
        rtdetr_cfg: dict,
        pickable_region_net_cfg: dict,
        losses_cfg: dict,
        is_distributed: bool = False,
        logger = None
    ):
        super(EventPickTargetPredictionNetwork, self).__init__()
        self._skip_rtdetr = self._skip_pickable_region = False

        self._concentration_net = ConcentrationNet(**concentration_net_cfg.PARAMS)
        freeze_module_grads(self._concentration_net)
        self._rtdetr = RTDETR(rtdetr_cfg=rtdetr_cfg)
        if rtdetr_cfg.is_freeze:
            print("rtdetr is frozen.")
            freeze_module_grads(self._rtdetr)
            self._skip_rtdetr = True
        self._pickable_region_predictor = PickableRegionNetwork(
            pickable_region_net_cfg.params["in_channels"],
            pickable_region_net_cfg.params["feat_channels"],
            pickable_region_net_cfg.params["num_classes"]
        )
        if pickable_region_net_cfg.is_freeze:
            print("pickable_region_net is frozen.")
            freeze_module_grads(self._pickable_region_predictor)
            self._skip_pickable_region = True
        self.bbox_roi_extractor = MODELS.build(
            {
                'type': 'SingleRoIExtractor',
                'roi_layer': {'type': 'RoIAlign', 'output_size': pickable_region_net_cfg.params['featmap_size'], 'sampling_ratio': 0},
                'out_channels': pickable_region_net_cfg.params['in_channels'],
                'featmap_strides': [1]
            }
        )

        # prepare the loss function
        matcher = HungarianMatcher(**losses_cfg.matcher.params)
        self.loss_functor = RTDETRCriterion(matcher, **losses_cfg.params)
        if not self._skip_pickable_region:
            self.loss_functor_pickableregion = torch.nn.SmoothL1Loss(reduction="mean")
        self.logger = logger

    def forward(self, left_event: Tensor, right_event: Tensor, gt_labels: Dict=None, batch_img_metas: Dict=None, global_step_info: Dict=None):
        results = {}
        list_event_sharp = []
        starttime = time.time()
        for x, side in zip([left_event, right_event], ["left", "right"]):
            x = x.squeeze(1).squeeze(3).permute(0, 3, 1, 2)
            sharp_repr = self._concentration_net(x)
            list_event_sharp.append(sharp_repr.detach().squeeze(1).cpu().numpy())
            results[side] = self._rtdetr(sharp_repr.repeat(1, 3, 1, 1), targets=gt_labels['objdet'][side] if gt_labels is not None else None)
            if gt_labels is None:
                results[side + "_sharprepr"] = sharp_repr.detach()
            else:
                bboxes_xyxy = torchvision.ops.box_convert(results[side]["pred_boxes"], in_fmt="cxcywh", out_fmt="xyxy".lower())
                bboxes_xyxy[..., [0, 2]] *= batch_img_metas['w']
                bboxes_xyxy[..., [1, 3]] *= batch_img_metas['h']            
                results[side + "_segMaps"] = self._predict_pickable_region(sharp_repr, bboxes_xyxy)
        if not self.training:
            print("bbox detection time cost: {} sec.".format(time.time() - starttime))

        loss_final = None
        if self.training:
            self.loss_functor.train()
        else:
            self.loss_functor.eval()

        if gt_labels is not None:
            assert (global_step_info is not None)
            epoch, lengthDataLoader, indexBatch = global_step_info["epoch"], global_step_info["lengthDataLoader"], global_step_info["indexBatch"]
            global_step = epoch * lengthDataLoader + indexBatch
            epoch_info = dict(epoch=epoch, step=indexBatch, global_step=global_step)
            losses_rtdetr = []
            losses_pickable_region = []
            predictions_leftright = {}
            for side in ["left", "right"]:
                loss, predictions, corresponding_gt_labels, indices = self.loss_functor(results[side], gt_labels["objdet"][side], **epoch_info)
                losses_rtdetr.extend(loss.values())
                if not self._skip_pickable_region:
                    loss_pickable_region, segmaps_forvisz = self._compute_loss_PickableRegionPerInstance(predictions, results[side + "_segMaps"], gt_labels["objdet"][side], indices, batch_img_metas)
                    losses_pickable_region.append(loss_pickable_region)
                    for indexPrediction in range(len(predictions)):
                        predictions[indexPrediction]["segMaps"] = segmaps_forvisz[indexPrediction]
                predictions_leftright[side] = predictions  # NOte: list, for batch inputs

            loss_final = {}
            loss_final["loss_rtdetr"] = sum(losses_rtdetr)
            if not self._skip_pickable_region:
                loss_final["loss_pickable_region"] = sum(losses_pickable_region) * 100

            if self.logger is not None and len(predictions_leftright) > 0:
                predictions_left = copy.deepcopy(predictions_leftright["left"])[0]
                predictions_left["bboxes"] = torchvision.ops.box_convert(predictions_left["bboxes"], in_fmt="cxcywh", out_fmt="xyxy".lower())
                predictions_left["bboxes"][:, [0, 2]] *= batch_img_metas['w']
                predictions_left["bboxes"][:, [1, 3]] *= batch_img_metas['h']
                view_img = RenderImageWithBboxes(list_event_sharp[0][0], predictions_left)
                self.logger.add_image(
                    "left sharp with bboxes",
                    view_img[0]
                )
        else:
            starttime = time.time()
            for side in ["left", "right"]:
                batch_size = results[side]["pred_logits"].shape[0]
                results[side + "_detections"] = []
                results[side + "_classes"] = []
                results[side + "_segMaps"] = []
                for indexInBatch in range(batch_size):                    
                    mask_pred_logits_objects = torch.max(results[side]["pred_logits"][indexInBatch], dim=-1)[0] > 0
                    results[side + "_detections"].append(results[side]["pred_boxes"][indexInBatch][mask_pred_logits_objects])
                    results[side + "_classes"].append(torch.round(torch.max(results[side]["pred_logits"][indexInBatch], dim=-1)[0][mask_pred_logits_objects]).to(torch.int))
                    results_bboxes_xyxy = self.convert_bboxes_to_xyxy(results[side + "_detections"][-1], batch_img_metas)
                    # compute pickable region per instance
                    pickable_region_predictions = self._predict_pickable_region(results[side + "_sharprepr"], results_bboxes_xyxy.unsqueeze(0)).squeeze()
                    instance_segmaps_one = []
                    for indexInstance in range(results_bboxes_xyxy.shape[0]):
                        instance_segmaps_one.append(pickable_region_predictions[indexInstance, results[side + "_classes"][-1][indexInstance]].unsqueeze(0))
                    if len(instance_segmaps_one):
                        results[side + "_segMaps"].append(torch.cat(instance_segmaps_one, dim=0).cpu().numpy())
            if not is_train:
                print("pickable region time cost: {} sec.".format(time.time() - starttime))

        return results, loss_final

    def _predict_pickable_region(self, events_repr: torch.tensor, bboxes: torch.tensor):
        """
        Args:
            events_repr: (b, 1, h, w). xyxy format with real h and w.
            bboxes: (b, N, 4). N is number of preset bboxes, the last channel is [tl_x, tl_y, br_x, br_y] format bbox.
        """
        batch_size, numPresetBboxes = bboxes.shape[:2]
        batch_number = torch.arange(bboxes.shape[0]).unsqueeze(1).expand(-1, numPresetBboxes).flatten().unsqueeze(-1).to(bboxes.device)
        rois = bboxes.reshape(-1, 4)
        bnum_rois = torch.cat([batch_number, rois], dim=1)
        roi_feats = self.bbox_roi_extractor([events_repr], bnum_rois)
        classes_feat_map = self._pickable_region_predictor(roi_feats)  # Note: (b*N, num_of_class, 28, 28)
        num_classes = classes_feat_map.shape[1]
        featmap_size = classes_feat_map.shape[-1]
        return classes_feat_map.view(batch_size, numPresetBboxes, num_classes, featmap_size, featmap_size).sigmoid()

    def _compute_loss_PickableRegionPerInstance(self, predicted_instances: list, predicted_prMap: torch.tensor, gt_labels: list, indices: list, batch_img_metas: dict):
        """
        Compute loss of pickable region prediction per instance.
        Args:
            predicted_instaces:
            predicted_prMap: predicted pickable region maps for each detected instances.
            gt_prMap: gt pickable region maps for each gt instances.
            indices: indices of the matched instances in the predefined number of instances.
        """
        batch_size = predicted_prMap.shape[0]
        gt_maps = []
        predicted_maps = []
        for indexInBatch in range(batch_size):
            indices_thisBatch = indices[indexInBatch]
            classes_thisBatch = gt_labels[indexInBatch]["labels"][indices_thisBatch[-1]]
            predicted_prMaps_selected = predicted_prMap[indexInBatch][indices_thisBatch[0]]
            predicted_prMaps_selected_classed = []
            for indexInstance in range(predicted_prMaps_selected.shape[0]):
                predicted_prMaps_selected_classed.append(predicted_prMaps_selected[indexInstance, classes_thisBatch[indexInstance]].unsqueeze(0))
            predicted_prMaps_selected_classed = torch.cat(predicted_prMaps_selected_classed, dim=0)

            predicted_bboxes = self.convert_bboxes_to_xyxy(predicted_instances[indexInBatch]["bboxes"], batch_img_metas)
            gt_bboxes = self.convert_bboxes_to_xyxy(gt_labels[indexInBatch]["boxes"][indices_thisBatch[-1]], batch_img_metas)
            batch_number = torch.ones((gt_bboxes.shape[0], 1), device = gt_bboxes.device) * indexInBatch
            bnum_rois = torch.cat([batch_number, gt_bboxes], dim=1)
            gt_prMaps_sorted = gt_labels[indexInBatch]["segMaps"][indices_thisBatch[-1]]
            gt_prMaps_instances = mask_target(
                [predicted_bboxes],
                [torch.arange(0, predicted_bboxes.shape[0])],
                [BitmapMasks(gt_prMaps_sorted.cpu().numpy(), height=batch_img_metas['h'], width=batch_img_metas['w'])],
                Config({"mask_size": predicted_prMaps_selected_classed.shape[-1], 'soft_mask_target': True}))

            if self.logger is not None:
                gt_feat = gt_prMaps_instances[0].detach().cpu()
                gt_feat = gt_feat - gt_feat.min()
                gt_feat *= 255 / gt_feat.max()
                # Note: half precision mode does not support F.interpolate
                # gt_feat = F.interpolate(gt_feat.unsqueeze(0).unsqueeze(0), size=(720, 720), mode='bilinear', align_corners=False)
                self.logger.add_image("pickable_region_GT_batch_{}_sample_0".format(indexInBatch), gt_feat.to(torch.uint8).squeeze())
                predicted_feat = predicted_prMaps_selected_classed[0].detach().cpu()
                predicted_feat = predicted_feat - predicted_feat.min()
                predicted_feat *= 255 / predicted_feat.max()
                # predicted_feat = F.interpolate(predicted_feat.unsqueeze(0).unsqueeze(0), size=(720, 720), mode='bilinear', align_corners=False)
                self.logger.add_image("pickable_region_predicted_batch_{}_sample_0".format(indexInBatch), predicted_feat.to(torch.uint8).squeeze())

            gt_maps.append(gt_prMaps_instances.unsqueeze(0))
            predicted_maps.append(predicted_prMaps_selected_classed.unsqueeze(0))
        gt_maps = torch.cat(gt_maps, dim=0)
        predicted_map_forvisz = [oneMap.detach().squeeze().cpu().numpy() for oneMap in predicted_maps]
        predicted_maps = torch.cat(predicted_maps, dim=0)
        return self.loss_functor_pickableregion(gt_maps.to(predicted_maps.device), predicted_maps), predicted_map_forvisz

    @staticmethod
    def convert_bboxes_to_xyxy(bboxes: torch.tensor, batch_img_metas: dict):
        """
        convert bboxes predicted by network (cxcywh, normalized) to xyxy format with real w and h.
        """
        bboxes = torchvision.ops.box_convert(bboxes, in_fmt="cxcywh", out_fmt="xyxy".lower())
        bboxes[:, [0, 2]] *= batch_img_metas["w"]
        bboxes[:, [1, 3]] *= batch_img_metas["h"]
        return bboxes

    @staticmethod
    def ComputeCostProfile(model: torch.nn.Module, inputShape: list):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        left_event = torch.randn(*inputShape).to(device)
        right_event = torch.randn(*inputShape).to(device)        
        model = model.to(device)
        flops, numParams = profile(model, inputs=(left_event, right_event, None, {'h': inputShape[2], 'w': inputShape[3]}), verbose=False)        
        return flops, numParams


class PickableRegionNetwork(nn.Module):
    def __init__(self, in_channels: int, feat_channels: int, num_classes: int):
        super(PickableRegionNetwork, self).__init__()
        stacked_convs = []
        in_channels_temp = in_channels
        for indexConvLayer in range(8):
            stacked_convs.append(
                nn.Conv2d(
                    in_channels_temp, feat_channels, kernel_size=3, stride=1, padding=1
                )
            )
            in_channels_temp = feat_channels
            stacked_convs.append(nn.ReLU(inplace=False))
        stacked_convs.append(
            nn.ConvTranspose2d(
                feat_channels, feat_channels, kernel_size=2, stride=2, padding=1
            )
        )
        stacked_convs.append(nn.ReLU(inplace=False))
        stacked_convs.append(
            nn.Conv2d(feat_channels, num_classes, kernel_size=1, stride=1, padding=1)
        )

        self._pickle_region_predictor = nn.Sequential(*stacked_convs)
        self._init_weights()

    def _init_weights(self):
        for subnet in [self._pickle_region_predictor]:
            for m in subnet.modules():
                if m is None:
                    continue
                elif hasattr(m, 'weight') and hasattr(m, 'bias'):
                    nn.init.kaiming_normal_(
                        m.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.constant_(m.bias, 0)

    def forward(self, feature_map: torch.tensor):
        return self._pickle_region_predictor(feature_map)
