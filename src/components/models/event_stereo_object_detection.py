import torch.nn as nn
import torch
from torch import Tensor
import numpy
from typing import List, Dict, Tuple
from thop import profile
import cv2
import time
import math

from mmdet.registry import MODELS

from .concentration import ConcentrationNet
from .stereo_matching import StereoMatchingNetwork
from .objectdetection import StereoEventDetectionHead



from . import losses
from .utils.misc import freeze_module_grads, unfreeze_module_grads, multi_apply, convert_tensor_to_numpy
from ..methods.visz_utils import RenderImageWithBboxes, RenderImageWithBboxesAndKeypts


class OnnxStyleNetwork(nn.Module):
    def __init__(self, model):
        super(OnnxStyleNetwork, self).__init__()
        self.model = model

    def forward(self, left_event: Tensor, right_event: Tensor, h_cam: Tensor, w_cam: Tensor):
        h_tensor, w_tensor = left_event.shape[-2:]
        batch_img_metas = {"h_cam": h_cam.item(), "w_cam": w_cam.item(), "h": h_tensor, "w": w_tensor}
        preds, losses = self.model(left_event, right_event, None, batch_img_metas)
        output_concentrate_left = preds["concentrate"]["left"]
        output_concentrate_right = preds["concentrate"]["right"]
        output_disparity = preds["disparity"][0]
        output_objdet = preds["objdet"][0]  # Note: should only use batch_size = 1 when doing inference.
        output_facets = preds["objdet_facets"][0]
        output_facets_right = preds["objdet_facets_right"][0]
        return output_objdet, output_facets, output_facets_right, output_concentrate_left, output_concentrate_right, output_disparity


class StereoDetectionHead(nn.Module):
    def __init__(
        self,
        network_cfg: dict,
        loss_cfg: dict,
        is_freeze: bool,
        **kwargs
    ):
        super(StereoDetectionHead, self).__init__()
        self._config = network_cfg
        self._config["is_freeze"] = is_freeze

        # objdet tools
        self.bbox_roi_extractor = MODELS.build(
            {
                'type': 'SingleRoIExtractor',
                'roi_layer': {'type': 'RoIAlign', 'output_size': self._config['right_roi_feat_size'], 'sampling_ratio': 0},
                'out_channels': self._config['in_channels'],
                'featmap_strides': [8, 16, 32]
            }
        )

        # loss
        self.loss_rbbox = MODELS.build({'type': 'IoULoss',  # Right side bbox
                                        'mode': 'square',
                                        'eps': 1e-16,
                                        'reduction': 'sum',
                                        'loss_weight': 5.0})
        self.loss_rscore = MODELS.build({'type': 'CrossEntropyLoss',
                                         'use_sigmoid': True,
                                         'reduction': 'sum',
                                         'loss_weight': 1.0})

        self._init_layers()
        self._init_weights()

    @property
    def is_freeze(self):
        return self._config["is_freeze"]
    
    @property
    def input_shape(self):
        return [(1, 10, 480, 672), (1, 10, 480, 672)]

    def _init_layers(self):
        self.right_bbox_refiner = self._build_bbox_refiner_convs(
            self._config['in_channels'],
            self._config['feat_channels'],
            output_logits=4,
            norm_eps=self._config["norm_cfg"]["eps"],
            norm_momentum=self._config["norm_cfg"]["momentum"],
            act_type=self._config["act_cfg"]["type"]
        )
        self.right_bbox_refiner_scorer = self._build_bbox_refiner_convs(
            self._config['in_channels'],
            self._config['feat_channels'],
            output_logits=1,
            norm_eps=self._config["norm_cfg"]["eps"],
            norm_momentum=self._config["norm_cfg"]["momentum"],
            act_type=self._config["act_cfg"]["type"]
        )

    def _init_weights(self):
        """
        all conv2d need weights initialization
        """
        for subnet in [self.right_bbox_refiner]:
            for module in subnet.modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    nn.init.kaiming_uniform_(
                        module.weight,
                        a=math.sqrt(5),
                        mode="fan_in",
                        nonlinearity="leaky_relu"
                    )

    @staticmethod
    def _build_bbox_refiner_convs(
        in_channels: int,
        feat_channels: int,
        output_logits: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        norm_eps: float = 1e-5,
        norm_momentum: float = 0.1,
        act_type: str = "ReLU"
    ) -> nn.Sequential:
        """
        For left, output logits are [delta_x, delta_y, w, h], w and h are relative values to bbox size;
        For right, output two logits are [delta_x, w].
        """
        bbox_refiner = nn.Sequential(
            nn.Conv2d(in_channels, feat_channels, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(feat_channels, eps=norm_eps, momentum=norm_momentum),
            getattr(nn, act_type)(),
            nn.Conv2d(feat_channels, feat_channels, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(feat_channels, eps=norm_eps, momentum=norm_momentum),
            getattr(nn, act_type)(),
            nn.Conv2d(feat_channels, output_logits, 1)
        )
        return bbox_refiner

    @staticmethod
    def ComputeCostProfile(model):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        input_tensor = torch.randn(*model.input_shape).to(device)
        model = model.to(device)
        flops, numParams = profile(model, inputs=input_tensor, verbose=False)
        return flops, numParams

    def predict(
        self,
        right_feat: Tensor,
        left_bboxes: List[Tensor],
        disp_prior: Tensor,
        batch_img_metas: Dict,
        bbox_expand_factor: float
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        """
        Args:
            right_feat: multi level features from left. max shape is [B, 3, h/8, w/8]
            bboxes_pred: list of B tensors of shape [?, 4]. [tl_x, tl_y, br_x, br_y] format bbox, all in global scale.
            disp_prior: [B, h, w] shape.
            bbox_expand_factor: ratio to enlarge bbox due to inaccurate disp prior.

        Returns:
            sbboxes_pred: shape [B, N, 6]. format [tl_x, tl_y, br_x, br_y, tl_x_r, br_x_r] rough stereo bbox
            refined_right_bboxes: shape [B, N, ker_h * ker_w, 4]. Corresponding refined right bboxes.
            right_scores_refine: shape [B, N, ker_h * ker_w, 1]. Corresponding scores.
        """
        list_sbboxes_pred, list_refined_right_bboxes, list_right_scores_refine = [], [], []
        for indexInBatch, bboxes_pred in enumerate(left_bboxes):
            starttime = time.time()
            bboxes_pred = torch.unsqueeze(bboxes_pred, dim=0)

            # enlarge bboxes
            _bboxes_pred = bboxes_pred.clone()  # initial warped bboxes for right side target.
            dwboxes = (_bboxes_pred[..., 2] - _bboxes_pred[..., 0]) * (bbox_expand_factor - 1.)
            # dhboxes = (_bboxes_pred[..., 3] - _bboxes_pred[..., 1]) * (bbox_expand_factor - 1.)
            _bboxes_pred[..., 0] -= dwboxes / 2  # Note: only need to enlarge x direction, because we assume for stereo y is aligned.
            _bboxes_pred[..., 2] += dwboxes / 2
            # _bboxes_pred[..., 1] -= dhboxes / 2
            # _bboxes_pred[..., 3] += dhboxes / 2

            batch_number = torch.arange(_bboxes_pred.shape[0]).unsqueeze(1).expand(-1, self._config['num_topk_candidates']).flatten().unsqueeze(-1).to(_bboxes_pred.device)
            # print("----- time sub sub1: {}".format(time.time() - starttime))

            starttime = time.time()
            # extract right bbox roi feature
            xindi = ((_bboxes_pred[..., 0] + _bboxes_pred[..., 2]) / 2).to(torch.int).clamp(0, batch_img_metas['w'] - 1)
            yindi = ((_bboxes_pred[..., 1] + _bboxes_pred[..., 3]) / 2).to(torch.int).clamp(0, batch_img_metas['h'] - 1)

            bbox_disps = disp_prior[indexInBatch][yindi[indexInBatch], xindi[indexInBatch]].unsqueeze(0)
            _bboxes_pred[..., 0] -= bbox_disps  # warp to more left side.
            _bboxes_pred[..., 2] -= bbox_disps
            rois_right = _bboxes_pred.reshape(-1, 4)
            rois_right = torch.cat((batch_number, rois_right), dim=1)        
            right_roi_feats = self.bbox_roi_extractor(right_feat, rois_right)  # Note: Based on the bbox size to decide from which level to extract feats.        
            right_boxes_refine = self.right_bbox_refiner(right_roi_feats)  # Note: shape [B*100, 4, 7, 7]
            right_scores_refine = self.right_bbox_refiner_scorer(right_roi_feats)  # Note: shape [B*100, 1, 7, 7]
            logits, ker_h, ker_w = right_boxes_refine.shape[-3:]
            # print("----- time sub sub2: {}".format(time.time() - starttime))

            # starttime = time.time()
            # if self.logger is not None:
            #     roi_feat_sample = right_roi_feats[0, 0, :, :].detach().cpu()
            #     roi_feat_sample = roi_feat_sample - roi_feat_sample.min()
            #     roi_feat_sample /= roi_feat_sample.max()
            #     self.logger.add_image("roi_feat_sample", roi_feat_sample)
            # print("----- time sub sub2.5: {}".format(time.time() - starttime))

            starttime = time.time()
            batch_size = bboxes_pred.shape[0]
            x_tl_r = _bboxes_pred[..., 0].view(batch_size, -1).unsqueeze(-1)
            x_br_r = _bboxes_pred[..., 2].view(batch_size, -1).unsqueeze(-1)
            sbboxes_pred = torch.cat([bboxes_pred, x_tl_r, x_br_r], dim=-1).contiguous()
            right_boxes_refine = right_boxes_refine.view(batch_size, -1, logits, ker_h, ker_w)
            refined_right_bboxes = self._right_bbox_decode(sbboxes_pred, right_boxes_refine)
            # print("----- time sub sub3: {}".format(time.time() - starttime))

            list_sbboxes_pred.append(sbboxes_pred)
            list_refined_right_bboxes.append(refined_right_bboxes)
            list_right_scores_refine.append(right_scores_refine.view(1, -1, 1, ker_h * ker_w).permute(0, 1, 3, 2))

        return list_sbboxes_pred, list_refined_right_bboxes, list_right_scores_refine

    def forward(
        self,
        right_feat: Tensor,
        left_bboxes: List[Tensor],
        disp_prior: Tensor,
        batch_img_metas: Dict,
        bbox_expand_factor: float,
        labels=None,
        **kwargs
    ):
        preds = self.predict(right_feat, left_bboxes, disp_prior, batch_img_metas, bbox_expand_factor)
        losses = None
        if labels is not None and not self.is_freeze:
            losses = self.compute_loss(preds, labels)
        artifacts = None
        return preds, losses, artifacts


class FeaturemapHead(nn.Module):
    def __init__(
        self,
        network_cfg: dict,
        loss_cfg: dict,
        is_freeze: bool,
        **kwargs
    ):
        super(FeaturemapHead, self).__init__()
        for key, config in network_cfg.items():
            if config.is_enable:
                self._config = network_cfg[key]
                self._config["pred_mode"] = key
                self._config["is_freeze"] = is_freeze
                break
        self._init_layers()
        self._init_weights()

    @property
    def is_freeze(self):
        return self._config["is_freeze"]

    @property
    def input_shape(self):
        return [(1, 10, 480, 672), (1, 10, 480, 672)]

    def _init_layers(self) -> None:
        if self._config["pred_mode"] == 'keypt_pred_cfg':
            # Note: two keypoints needed. One at object center, one at object top center.
            self.keypt1_predictor = self._build_featmap_convs(
                in_channels=self._config.PARAMS['keypt_in_channels'],
                feat_channels=self._config.PARAMS['keypt_feat_channels'],
                num_classes=self._config.PARAMS["num_classes"],
            )
            self.keypt2_predictor = self._build_featmap_convs(
                in_channels=self._config.PARAMS['keypt_in_channels'],
                feat_channels=self._config.PARAMS['keypt_feat_channels'],
                num_classes=self._config.PARAMS["num_classes"],
            )
        elif self._config["pred_mode"] == 'facet_pred_cfg':
            self.facet_predictor = self._build_featmap_convs(
                in_channels=self._config.PARAMS['facet_in_channels'],
                feat_channels=self._config.PARAMS['facet_feat_channels'],
                num_classes=self._config.PARAMS['num_classes'],
            )

    def _init_weights(self):
        if self.keypt1_predictor is not None:
            for subnet in [self.keypt1_predictor, self.keypt2_predictor]:
                for m in subnet.modules():
                    if m is None:
                        continue
                    elif hasattr(m, 'weight') and hasattr(m, 'bias'):
                        nn.init.kaiming_normal_(
                            m.weight, mode='fan_out', nonlinearity='relu')
                        nn.init.constant_(m.bias, 0)
        elif self.facet_predictor is not None:
            for m in self.facet_predictor.modules():
                if m is None:
                    continue
                elif hasattr(m, 'weight') and hasattr(m, 'bias'):
                    nn.init.kaiming_normal_(
                        m.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.constant_(m.bias, 0)

    @staticmethod
    def _build_featmap_convs(in_channels: int, feat_channels: int, num_classes: int):
        """
        refer to mask-rcnn mask-prediction head.
        """
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

        return nn.Sequential(*stacked_convs)

    @staticmethod
    def ComputeCostProfile(model):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        input_tensor = torch.randn(*model.input_shape).to(device)
        model = model.to(device)
        flops, numParams = profile(model, inputs=input_tensor, verbose=False)
        return flops, numParams

    def predict(
        self,
        img_feats: List[Tensor],
        bbox_preds: List[Tensor],
        featuremap_predictor: nn.Sequential,
        feature_id: str
    ) -> Tensor:
        """
        bbox_pred is top 100 bbox predictions from forward_objdet_single. Same as mask-rcnn.
        predict keypoint bias to the grid center, e.g. [delta_x, delta_y].
        predict one keypoint for each class, shape of one output is (1, ?, num_classes, 2). only take the k-th keypoint, k is the index of class of the bbox.

        Args:
            img_feats: multi level features.
            bbox_preds: B of (?, 4) shape tensor. predicted bbox of objects in the batch. ? is the number of bboxes. 4 channels are [tl_x, tl_y, br_x, br_y] format bbox.

        Return:
            list_keypts_pred: B of (1, 100, num_classes, 2) tensor. Predicted keypoint bias (normalized) to the bbox top left corner.
        """
        list_featmap_preds = []
        for indexInBatch in range(len(bbox_preds)):
            bbox_pred = bbox_preds[indexInBatch].unsqueeze(0)
            batch_number = torch.arange(bbox_pred.shape[0]).unsqueeze(1).expand(-1, bbox_pred.shape[1]).flatten().unsqueeze(-1).to(bbox_pred.device)
            with torch.no_grad():
                rois = bbox_pred.reshape(-1, 4)
                enlarge_factor = self._config.PARAMS["enlarge_roi_factor"]
                rois_w = (rois[:, 2] - rois[:, 0]) * enlarge_factor
                rois_h = (rois[:, 3] - rois[:, 1]) * enlarge_factor
                rois_centerX = (rois[:, 2] + rois[:, 0]) / 2
                rois_centerY = (rois[:, 3] + rois[:, 1]) / 2
                rois[:, 0] = rois_centerX - rois_w / 2
                rois[:, 1] = rois_centerY - rois_h / 2
                rois[:, 2] = rois_centerX + rois_w / 2
                rois[:, 3] = rois_centerY + rois_h / 2
            bnum_rois = torch.cat([batch_number, rois], dim=1)
            roi_feats = self.bbox_roi_extractor_forfeature([onefeat[indexInBatch].unsqueeze(0) for onefeat in img_feats], bnum_rois)  # Note: output shape is (b*100, 128, 14, 14)
            output_feat = featuremap_predictor(roi_feats)  # Note: (b*100, num_of_class, 28, 28)
            # if self.logger is not None:
            #     self.feat_roi_visz[feature_id] = roi_feats.detach()
            #     self.feat_output_visz[feature_id] = output_feat.detach()
            num_bboxes = bbox_pred.shape[1]
            num_classes = self._config.PARAMS['num_classes']
            featmap_size = output_feat.shape[-1]
            list_featmap_preds.append(output_feat.view(1, num_bboxes, num_classes, featmap_size, featmap_size).sigmoid())
        return list_featmap_preds

    def forward(
        self,
        img_feats: List[Tensor],
        bbox_preds: List[Tensor],
        featuremap_predictor: nn.Sequential,
        feature_id: str,
        labels=None,
        **kwargs
    ):
        preds = self.predict(img_feats, bbox_preds, featuremap_predictor, feature_id)
        losses = None
        if labels is not None and not self.is_freeze:
            losses = self.compute_loss(preds, labels)
        artifacts = None
        return preds, losses, artifacts


class EventStereoObjectDetectionNetwork(nn.Module):

    def __init__(
        self,
        concentration_net_cfg: dict = None,
        disp_head_cfg: dict = None,
        object_detection_head_cfg: dict = None,
        losses_cfg: dict = None,  # StereoEventDetectionLoss, disparityLoss
        is_distributed: bool = False,
        is_test = False,
        logger=None,
    ):
        super(EventStereoObjectDetectionNetwork, self).__init__()
        self.is_test = False  # Note: whether is test only
        self.logger = logger
        self.is_freeze_disp = disp_head_cfg['is_freeze']  # Note: when training disparity, skip object detection.
        is_train_featmaponly = (
                object_detection_head_cfg['PARAMS']['keypt_pred_cfg']['is_enable'] and object_detection_head_cfg['PARAMS']['keypt_pred_cfg']['PARAMS']['is_train_keypt']
            ) or (
                object_detection_head_cfg['PARAMS']['facet_pred_cfg']['is_enable'] and object_detection_head_cfg['PARAMS']['facet_pred_cfg']['PARAMS']['is_train_facet']
            )
        # ==========  concentration net ===========
        self._concentration_net = ConcentrationNet(**concentration_net_cfg.PARAMS)
        if self.is_freeze_disp:
            freeze_module_grads(self._concentration_net)
        # ============ stereo matching net ============
        self._disp_head = StereoMatchingNetwork(
            **disp_head_cfg.PARAMS, isInputFeature=False  # Note: an efficient feature extractor for object detection might not be good for stereo matching?
        )
        if self.is_freeze_disp:
            freeze_module_grads(self._disp_head)
        # ============= object detection net =============
        self._object_detection_head = StereoEventDetectionHead(
            net_cfg=object_detection_head_cfg.PARAMS,
            loss_cfg=losses_cfg,
            is_distributed=is_distributed,
            logger=logger
        )
        if not self.is_freeze_disp:
            freeze_module_grads(self._object_detection_head)
        # ============= losses ============
        if not self.is_freeze_disp:
            self._disp_loss = getattr(losses, losses_cfg['disp_loss_cfg']['NAME'])(
                losses_cfg['disp_loss_cfg'],
                is_distributed=is_distributed,
                logger=logger
            )

    def SetTest(self):
        self.is_test = True
    
    def SetNotTest(self):
        self.is_test = False

    def forward(self, left_event: Tensor, right_event: Tensor, gt_labels: dict, batch_img_metas: dict = None, global_step_info: dict = None, **kwargs):
        """
        Args:
            left/right_event: (b c s t h w) tensor
            gt_labels: a dict, each dict contains 'disparity' and 'objdet'.
                       'disparity' is a Tensor; 'objdet' is list of dict, each dict contains 'bboxes' and 'labels' and 'keypt_masks'.
            batch_img_metas: dict. It contains info about image height and width.
        """
        if self.logger is not None:
            event_view = left_event[0, -3, :, :].detach().cpu()
            event_view -= event_view.min()
            event_view /= event_view.max()
            self.logger.add_image(
                "left event input",
                event_view
            )

        starttime = time.time()        
        left_event_sharp = self._concentration_net(left_event)
        right_event_sharp = self._concentration_net(right_event)
        # print("time1: {}".format(time.time() - starttime))

        # FIXME: test using same feature extractor for both head
        starttime = time.time()
        pred_disparity_pyramid = self._disp_head(left_event_sharp, right_event_sharp)
        # print("time2: {}".format(time.time() - starttime))

        loss_final = None
        if not self.is_freeze_disp and len(gt_labels) > 0:
            loss_final = self._disp_loss((
                pred_disparity_pyramid,
                gt_labels['disparity'],
                left_event_sharp,
                right_event_sharp
            ))

        if self.is_freeze_disp or len(gt_labels) == 0:
            starttime = time.time()
            object_preds, loss_final = self._object_detection_head(
                left_event,
                right_event,
                pred_disparity_pyramid[-1],  # use full size disparity prediction as prior to help stereo detection
                batch_img_metas,
                gt_labels["objdet"] if gt_labels is not None and len(gt_labels) != 0 else None,
                global_step_info
            )
            # print("time4: {}".format(time.time() - starttime))

            if self.logger is not None and len(object_preds) > 0 and loss_final is not None and (not self.is_test):
                if 'sbboxes' in object_preds[0]:
                    leftimage_views, rightimage_views = multi_apply(
                        RenderImageWithBboxesAndKeypts,
                        left_event_sharp.detach().squeeze(1).cpu().numpy(),
                        right_event_sharp.detach().squeeze(1).cpu().numpy(),
                        convert_tensor_to_numpy(object_preds)
                    )
                    self.logger.add_image(
                        "left sharp with bboxes",
                        leftimage_views[0]
                    )
                    self.logger.add_image(
                        "right sharp with bboxes",
                        rightimage_views[0]
                    )

                    # gt_objects_preds = []
                    # for gt_objects in gt_labels["objdet"]:
                    #     gt_objects_preds.append({
                    #         "sbboxes": gt_objects["bboxes"][:, :6].cpu(),
                    #         "classes": gt_objects["labels"].cpu(),
                    #         "confidences": torch.ones((gt_objects["labels"].shape[0],), dtype=gt_objects["labels"].dtype, device='cpu')
                    #     })
                    # leftimage_withGT_views, rightimage_withGT_views = multi_apply(
                    #     RenderImageWithBboxesAndKeypts,
                    #     left_event_sharp.detach().squeeze(1).cpu().numpy(),
                    #     right_event_sharp.detach().squeeze(1).cpu().numpy(),
                    #     gt_objects_preds
                    # )
                    # self.logger.add_image(
                    #     "left sharp with GT bboxes",
                    #     leftimage_withGT_views[0]
                    # )
                    # self.logger.add_image(
                    #     "right sharp with GT bboxes",
                    #     rightimage_withGT_views[0]
                    # )
                else:
                    leftimage_views = []
                    num_imgs = left_event.shape[0]
                    for i in range(num_imgs):
                        leftimage_views.append(
                            RenderImageWithBboxes(left_event_sharp.detach().squeeze(1).cpu().numpy()[i], object_preds[i])
                        )
                    self.logger.add_image(
                        "left sharp with bboxes, keypoints",
                        leftimage_views[0]
                    )
            if len(object_preds) == 0:
                print("Zero detection occured.")

        # Prepare inference output.
        preds_final = {}
        # split objdet and objdet_facets if objdet_facets exist.
        if self.is_test:
            if len(object_preds) == 0:
                preds_final['objdet'] = object_preds
            elif isinstance(object_preds[0], torch.Tensor):
                preds_final['objdet'] = [pred.detach().cpu() for pred in object_preds]
            elif isinstance(object_preds[0], dict):
                preds_final['objdet_facets'] = [detection.pop('facets').detach().cpu() for detection in object_preds]
                preds_final['objdet_facets_right'] = [detection.pop('facets_right').detach().cpu() for detection in object_preds]
                preds_final['enlarge_facet_factor'] = [detection.pop('enlarge_facet_factor') for detection in object_preds]
                preds_final['objdet'] = [detection.pop('detection').detach().cpu() for detection in object_preds]
            preds_final['concentrate'] = {
                'left': left_event_sharp.detach().cpu(),
                'right': right_event_sharp.detach().cpu()
            }
        preds_final['disparity'] = pred_disparity_pyramid[-1].detach().cpu()

        torch.cuda.synchronize()

        if self.logger is not None:
            lsharp_view = left_event_sharp[0, 0, :, :,].detach().cpu()
            lsharp_view -= lsharp_view.min()
            lsharp_view /= lsharp_view.max()
            self.logger.add_image(
                "left sharp",
                lsharp_view
            )
            disparity_view = pred_disparity_pyramid[-1].detach().cpu()
            disparity_view -= disparity_view.min()
            disparity_view /= disparity_view.max()
            self.logger.add_image(
                "disparity view",
                disparity_view
            )
            if gt_labels is not None and 'disparity' in gt_labels:
                disparity_gt = gt_labels['disparity'].detach().squeeze().cpu()
                disparity_gt -= disparity_gt.min()
                disparity_gt /= disparity_gt.max()
                self.logger.add_image(
                    "disparity gt",
                    disparity_gt
                )

        return preds_final, loss_final

    def get_params_group(self, learning_rate, keypt_lr=None):
        if keypt_lr is not None:
            specific_layer_name = ['_keypt_feature_extraction_net', 'keypt2_predictor', 'keypt1_predictor', "offset_conv.weight", "offset_conv.bias"]
        else:
            specific_layer_name = ["offset_conv.weight", "offset_conv.bias"]# Note: exist in deform conv.
        def filter_specific_params(kv):
            specific_layer_name = ["offset_conv.weight", "offset_conv.bias"]  
            for name in specific_layer_name:
                if name in kv[0]:
                    return True
            return False

        keypt_layer_name = ["keypt1_predictor", "keypt2_predictor"]
        def filter_keypt_params(kv):
            for name in keypt_layer_name:
                if name in kv[0]:
                    return True
            return False

        all_specific_layer_name = ["offset_conv.weight", "offset_conv.bias"]
        if keypt_lr is not None:
            all_specific_layer_name.extend(keypt_layer_name)
        def filter_base_params(kv):
            for name in all_specific_layer_name:
                if name in kv[0]:
                    return False
            return True

        specific_params = list(filter(filter_specific_params, self.named_parameters()))
        base_params = list(filter(filter_base_params, self.named_parameters()))

        specific_params = [
            kv[1] for kv in specific_params
        ]  # kv is a tuple (key, value)
        base_params = [kv[1] for kv in base_params]

        specific_lr = learning_rate * 0.1
        if keypt_lr is None:
            params_group = [
                {"params": base_params, "lr": learning_rate},
                {"params": specific_params, "lr": specific_lr},
            ]
        else:
            keypt_params = list(filter(filter_keypt_params, self.named_parameters()))
            keypt_params = [kv[1] for kv in keypt_params]
            params_group = [
                {"params": base_params, "lr": learning_rate},
                {"params": specific_params, "lr": specific_lr},
                {"params": keypt_params, "lr": keypt_lr}
            ]   
        return params_group

    @staticmethod
    def ComputeCostProfile(model, inputShape):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        left_event = torch.randn(*inputShape).to(device)
        right_event = torch.randn(*inputShape).to(device)
        model = model.to(device)
        global_step_info = dict(epoch=0, indexBatch=0, lengthDataLoader=0)
        flops, numParams = profile(model, inputs=(left_event, right_event, {}, {'h': inputShape[-2], 'w': inputShape[-1]}, global_step_info), verbose=False)
        return flops, numParams

