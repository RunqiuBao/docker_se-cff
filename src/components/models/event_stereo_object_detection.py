import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import Tensor
import numpy
from typing import List, Dict, Tuple, Optional
from thop import profile
import cv2
import time
import math

from mmdet.registry import MODELS
from mmengine.config import Config
from mmdet.structures.mask import mask_target, BitmapMasks

from .concentration import ConcentrationNet
from .stereo_matching import StereoMatchingNetwork
from .yolo_pose_blocks import Conv
from .yolo_pose_utils import make_anchors
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

        # keypoint prediction
        self._anchors = None
        self._strides = None
        self.loss_keypoint = None
        kpt_shape = (2, 3)
        nk = kpt_shape[0] * kpt_shape[1]  # number of keypoints total
        feat_channels = (128, 128, 128)
        c4 = max(feat_channels[0] // 4, nk)
        self.keypts_predictor = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, nk, 1)) for x in feat_channels)
        self._config["keypts_prediction"] = {
            "kpt_shape": kpt_shape,
            "nk": nk,
            "feat_channels": feat_channels,
            "stride": (8.0, 16.0, 32.0)
        }
        self._config["loss_cfg"] = loss_cfg

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
        self.loss_bce_pose = nn.BCEWithLogitsLoss()


        self._init_layers()
        self._init_weights()

    @property
    def is_freeze(self):
        return self._config["is_freeze"]
    
    @property
    def config(self):
        return self._config

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
        h, w = 480, 672
        input_feats = [
            torch.randn(4, 128, int(h / 8), int(w / 8)).to(device),
            torch.randn(4, 128, int(h / 16), int(w / 16)).to(device),
            torch.randn(4, 128, int(h / 32), int(w / 32)).to(device)
        ]
        bboxes = [torch.randn(120, 4).to(device)]
        disp_prior = torch.randn(4, h, w).to(device)
        batch_img_metas = {"h": h, "w": w}
        model = model.to(device)
        flops, numParams = profile(model, inputs=(input_feats,  bboxes, disp_prior, batch_img_metas), verbose=False)
        return flops, numParams
    
    def _right_bbox_decode(self, sbboxes_pred: Tensor, right_boxes_refine: Tensor) -> Tensor:
        """
        Decode right bbox refine result [B, 100, 4, ker_h, ker_w] whose '4' dimension is (delta_x, delta_y, w_factor, h_factor) to
        same shape whose '4' dimension is (tl_x_r, tl_y_r, br_x_r, br_y_r).
        Args:
            sbboxes_pred:  [B, 100, 6]
            right_boxes_refine: [B, 100, 4, ker_h, ker_w]
        
        Return:
            decoded_right_bboxes: [B, 100, ker_h, ker_w, 4] whose '4' dimension is (tl_x_r, tl_y_r, br_x_r, br_y_r)
        """
        batch_size, num_samples = sbboxes_pred.shape[:2]
        ker_h, ker_w = right_boxes_refine.shape[-2:]
        sbboxes_pred = sbboxes_pred.view(-1, 6)
        right_boxes_refine = right_boxes_refine.view(-1, 4, ker_h, ker_w)
        
        strides_x = (sbboxes_pred[:, 5] - sbboxes_pred[:, 4]) / ker_w
        strides_y = (sbboxes_pred[:, 3] - sbboxes_pred[:, 1]) / ker_h
        strides_x = strides_x.view(-1, 1, 1).expand(-1, ker_h, ker_w).unsqueeze(1)
        strides_y = strides_y.view(-1, 1, 1).expand(-1, ker_h, ker_w).unsqueeze(1)
        grid_x = torch.arange(0, ker_w, device=sbboxes_pred.device, dtype=sbboxes_pred.dtype).view(1, 1, -1).expand(batch_size * num_samples, ker_h, -1) * strides_x.squeeze(1)
        grid_x += sbboxes_pred[:, 4].clone().view(-1, 1, 1).expand(-1, ker_h, ker_w)
        grid_x = grid_x.unsqueeze(1)
        grid_y = torch.arange(0, ker_h, device=sbboxes_pred.device, dtype=sbboxes_pred.dtype).view(1, -1, 1).expand(batch_size * num_samples, -1, ker_w) * strides_y.squeeze(1)
        grid_y += sbboxes_pred[:, 1].clone().view(-1, 1, 1).expand(-1, ker_h, ker_w)
        grid_y = grid_y.unsqueeze(1)

        strides = torch.cat([strides_x, strides_y], dim=1)  # [B*100, 2, ker_h, ker_w]
        grids = torch.cat([grid_x, grid_y], dim=1)  # [B*100, 2, ker_h, ker_w]
        xys = right_boxes_refine[:, :2, :, :] * strides + grids
        whs = right_boxes_refine[:, 2:, :, :].exp() * strides

        tl_x = (xys[:, 0, :, :] - whs[:, 0, :, :] / 2).unsqueeze(-1)
        tl_y = (xys[:, 1, :, :] - whs[:, 1, :, :] / 2).unsqueeze(-1)
        br_x = (xys[:, 0, :, :] + whs[:, 0, :, :] / 2).unsqueeze(-1)
        br_y = (xys[:, 1, :, :] + whs[:, 1, :, :] / 2).unsqueeze(-1)

        decoded_right_bboxes = torch.cat([tl_x, tl_y, br_x, br_y], dim=-1)
        return decoded_right_bboxes.view(batch_size, num_samples, ker_h * ker_w, 4)

    def predict(
        self,
        right_feat: Tensor,
        left_bboxes: List[Tensor],
        disp_prior: Tensor,
        batch_img_metas: Dict,
        bbox_expand_factor: float,
        decode_keypts: bool
    ) -> Tuple[List[Optional[Tensor]], List[Optional[Tensor]], List[Optional[Tensor]]]:
        """
        Args:
            right_feat: multi level features from left. max shape is [B, 3, h/8, w/8]
            bboxes_pred: list of B tensors of shape [?, 4]. [tl_x, tl_y, br_x, br_y] format bbox, all in global scale.
            disp_prior: [B, h, w] shape.
            bbox_expand_factor: ratio to enlarge bbox due to inaccurate disp prior.
            decode_keypts: whether to decode the keypts prediction in the end.

        Returns:
            sbboxes_pred: shape [B, N, 6]. format [tl_x, tl_y, br_x, br_y, tl_x_r, br_x_r] rough stereo bbox
            refined_right_bboxes: shape [B, N, ker_h * ker_w, 4]. Corresponding refined right bboxes.
            right_scores_refine: shape [B, N, ker_h * ker_w, 1]. Corresponding scores.
        """
        list_sbboxes_pred, list_refined_right_bboxes, list_right_scores_refine = [], [], []
        for indexInBatch, bboxes_pred in enumerate(left_bboxes):
            if bboxes_pred.shape[0] == 0:
                # No detections in left
                list_sbboxes_pred.append(None)
                list_refined_right_bboxes.append(None)
                list_right_scores_refine.append(None)
                continue
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

            num_detections = bboxes_pred.shape[1]
            batch_number = torch.arange(_bboxes_pred.shape[0]).unsqueeze(1).expand(-1, num_detections).flatten().unsqueeze(-1).to(_bboxes_pred.device)
            # print("----- time sub sub1: {}".format(time.time() - starttime))

            starttime = time.time()
            # extract right bbox roi feature
            xindi = ((_bboxes_pred[..., 0] + _bboxes_pred[..., 2]) / 2).to(torch.int).clamp(0, batch_img_metas['w'] - 1).squeeze()
            yindi = ((_bboxes_pred[..., 1] + _bboxes_pred[..., 3]) / 2).to(torch.int).clamp(0, batch_img_metas['h'] - 1).squeeze()
            bbox_disps = disp_prior[indexInBatch][yindi, xindi].unsqueeze(0)
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

        # keypoints prediction
        batch_size = len(left_bboxes)
        pred_kpts = torch.cat([self.keypts_predictor[i](right_feat[i]).view(batch_size, self._config["keypts_prediction"]["nk"], -1) for i in range(len(self._config["keypts_prediction"]["feat_channels"]))], -1)

        # prepare data for keypoint decode. update anchors with new disp for every batch
        disp_multilevel = [F.interpolate(disp_prior.unsqueeze(1), scale_factor=(1 / scale_factor), mode='area') / scale_factor for scale_factor in self._config["keypts_prediction"]["stride"]]
        disp_multilevel = [disp.squeeze() for disp in disp_multilevel]
        self._anchors, self._strides = make_anchors(right_feat, self._config["keypts_prediction"]["stride"], 0.5, disp_prior=disp_multilevel)

        if decode_keypts:
            pred_kpts = self.decode_rescale_kpts(pred_kpts)

        return list_sbboxes_pred, list_refined_right_bboxes, list_right_scores_refine, pred_kpts

    def decode_rescale_kpts(self, pred_kpts: Tensor):
        ndim = self._config["keypts_prediction"]["kpt_shape"][1]
        y = pred_kpts.clone()
        y = y.permute(0, 2, 1)
        if ndim == 3:
            y[..., 2::3] = y[..., 2::3].sigmoid()        
        y[..., 0::ndim] = (y[..., 0::ndim] * 2.0 + (self._anchors[..., 0] - 0.5).unsqueeze(-1).expand(-1, -1, 2)) * self._strides.expand(-1, -1, 2)
        y[..., 1::ndim] = (y[..., 1::ndim] * 2.0 + (self._anchors[..., 1] - 0.5).unsqueeze(-1).expand(-1, -1, 2)) * self._strides.expand(-1, -1, 2)
        return y

    def forward(
        self,
        right_feat: List[Tensor],
        left_bboxes: List[Tensor],
        disp_prior: Tensor,
        batch_img_metas: Dict,
        labels=None,
        **kwargs
    ):
        """
        Note that gt bboxes in labels should align with left_bboxes and have same number (left detections are from detr).
        """
        decode_keypts = False
        if "decode_keypts" in kwargs:
            decode_keypts = kwargs["decode_keypts"]
        preds = self.predict(right_feat, left_bboxes, disp_prior, batch_img_metas, self._config["bbox_expand_factor"], decode_keypts)
        losses = None
        artifacts = None
        if labels is not None and not self.is_freeze:
            if kwargs["detector_format"] == "detr":
                losses = self.compute_loss_detrformat(preds, labels)
            elif kwargs["detector_format"] == "yolox":
                losses, artifacts = self.compute_loss_yoloxformat(preds, labels)
            elif kwargs["detector_format"] == "yolopose":
                losses, sbboxes, masks, selected_keypts = self.compute_loss_yoloposeformat(preds, labels)
                artifacts = [sbboxes, masks, selected_keypts]
        return preds, losses, artifacts
    
    
    def compute_loss_yoloposeformat(self, preds: Tuple[List, List, List], labels: Dict):
        """
        Args:
            preds:
            labels: dict containing the following keys.
                left_fg_mask: mask for target assignment. (B, max_hypotheses)-> (4, 6615).
                left_target_gt_idx: for each foreground object, idx to the corresponding gt target. (B, max_hypotheses). need to be selected by fg_mask before use.
                left_target_bboxes: all target bboxes from left side. (B, max_hypotheses, 4). used for loss computation in keypoints prediction.
                left_nms_topk_mask: mask of filtering noisy left detections by nms and topk. (B, max_hypotheses).
                batch_img_metas: h, w of img.
                stereo_objdet_targets: dict
                    "bboxes": (N_allbatch, 6),
                    "cls":  (N_allbatch,),
                    "keypoints_right": (N_allbatch, 2, 3),
                    "batch_idx": (N_allbatch,). Index in this batch for this instance.
        """
        list_sbboxes_pred, list_refined_right_bboxes, list_right_scores_refine, pred_kpts = preds  # Note: number of positive in each can be different after nms.
        left_fg_mask, left_target_gt_idx, left_nms_topk_mask, stereo_objdet_targets, batch_img_metas = labels["left_fg_mask"], labels["left_target_gt_idx"], labels["left_nms_topk_mask"], labels["stereo_objdet_targets"], labels["batch_img_metas"]

        loss_dict = {}
        num_batch = len(list_sbboxes_pred)
        list_sbboxes_pred_refined = []
        bbox_targets = []
        pos_masks = []
        # right bboxes related loss
        for indexInBatch in range(num_batch):
            bbox_targetset_one = stereo_objdet_targets["bboxes"][stereo_objdet_targets["batch_idx"] == indexInBatch]
            bbox_targets_one = bbox_targetset_one[left_target_gt_idx[indexInBatch][left_nms_topk_mask[indexInBatch]]]
            pos_masks_one = left_fg_mask[indexInBatch][left_nms_topk_mask[indexInBatch]]

            rbboxes_targets = torch.concat([
                bbox_targets_one[:, 4].unsqueeze(-1),
                bbox_targets_one[:, 1].unsqueeze(-1),
                bbox_targets_one[:, 5].unsqueeze(-1),
                bbox_targets_one[:, 3].unsqueeze(-1)
            ], dim=1)[pos_masks_one]
            right_bboxes = list_refined_right_bboxes[indexInBatch]
            right_scores = list_right_scores_refine[indexInBatch]
            bboxes = list_sbboxes_pred[indexInBatch]  # shape [1, 100, 6]. (tl_x, tl_y, br_x, br_y, tl_x_r, br_x_r) format bbox, all in global scale.

            batch_size, num_priors, num_grids = right_bboxes.shape[:3]
            rbboxes_refined = right_bboxes.view(-1, num_grids, 4)[pos_masks_one]
            num_positive = rbboxes_refined.shape[0]
            ious, indicies_best_right = self.batch_iou_calculator_simple(rbboxes_refined, rbboxes_targets.unsqueeze(1))

            rselect_mask, rbboxes_refined_selected, rbboxes_targets_selected = self.batch_assigner(
                rbboxes_refined,
                ious,
                rbboxes_targets,
                self._config['r_iou_threshold'],
                self._config['candidates_k']
            )
            num_pos_timesk = torch.sum(rselect_mask.to(torch.float))
            num_total_samples_timesk = max(num_pos_timesk, 1.0)
            
            loss_rbbox_one = self.loss_rbbox(rbboxes_refined_selected, rbboxes_targets_selected) / num_total_samples_timesk
            if "loss_rbbox" not in loss_dict:
                loss_dict["loss_rbbox"] = loss_rbbox_one
            else:
                loss_dict["loss_rbbox"] += loss_rbbox_one

            # right scores
            rbboxes_scores = right_scores.view(-1, num_grids, 1)[pos_masks_one].sigmoid()
            ## dynamic targets
            rbboxes_scores_targets = torch.zeros_like(rbboxes_scores)
            rbboxes_scores_targets[rselect_mask] = 1
            
            loss_rscore_one = self.loss_rscore(rbboxes_scores, rbboxes_scores_targets.view(num_positive, -1, 1)) / num_total_samples_timesk
            if "loss_rscore" not in loss_dict:
                loss_dict["loss_rscore"] = loss_rscore_one
            else:
                loss_dict["loss_rscore"] += loss_rscore_one

            # substitute right bboxes in sbboxes for visualization.
            bboxes = bboxes.view(-1, 6)
            selected_sbboxes = bboxes[pos_masks_one]
            indices_highest_score = torch.argmax(rbboxes_scores.squeeze(-1), dim=1)
            rbboxes_highest_score = torch.gather(rbboxes_refined, 1, indices_highest_score.view(num_positive, 1, 1).expand(-1, -1, 4)).squeeze(1)            
            selected_sbboxes[:, 4] = rbboxes_highest_score[:, 0]
            selected_sbboxes[:, 5] = rbboxes_highest_score[:, 2]
            bboxes[pos_masks_one] = selected_sbboxes
            bboxes = bboxes.view(num_priors, 6)
            list_sbboxes_pred_refined.append(bboxes)
            pos_masks.append(pos_masks_one)
            # print("----- time sub sub2 loss stereo: {}".format(time.time() - starttime))
            
        loss_dict["loss_rbbox"] /= num_batch
        loss_dict["loss_rscore"] /= num_batch

        # keypoints related loss
        batch_rbboxes_targets = stereo_objdet_targets['bboxes'][:, :6]
        batch_rbboxes_targets = torch.concat([
                batch_rbboxes_targets[:, 4].unsqueeze(-1),
                batch_rbboxes_targets[:, 1].unsqueeze(-1),
                batch_rbboxes_targets[:, 5].unsqueeze(-1),
                batch_rbboxes_targets[:, 3].unsqueeze(-1)
            ], dim=1)
        batch_rbboxes_targets = batch_rbboxes_targets[left_target_gt_idx]
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()
        pred_kpts = losses.kpts_decode(self._anchors, pred_kpts.view(num_batch, -1, *self._config["keypts_prediction"]["kpt_shape"]))
        gt_keypoints = stereo_objdet_targets["keypoints_right"]
        gt_keypoints[..., 0] *= batch_img_metas["w"]
        gt_keypoints[..., 1] *= batch_img_metas["h"]
        if self.loss_keypoint is None:
            OKS_SIGMA = (
                numpy.array([0.26, 0.25])  # Note: hard-coded values
                / 10.0
            )
            sigmas = torch.from_numpy(OKS_SIGMA).to(pred_kpts.device)
            self.loss_keypoint = losses.KeypointLoss(sigmas=sigmas)
        loss_kptpose, loss_kptobj = losses.calculate_keypoints_loss(
            left_fg_mask,
            left_target_gt_idx,
            gt_keypoints,
            stereo_objdet_targets["batch_idx"],
            self._strides,
            batch_rbboxes_targets,
            pred_kpts,
            self.loss_bce_pose,
            self.loss_keypoint
        )
        loss_dict["loss_kptpose"] = loss_kptpose * self._config["loss_cfg"]["pose_loss_weight"]
        loss_dict["loss_kptobj"] = loss_kptobj * self._config["loss_cfg"]["kobj_loss_weight"]
        pred_kpts_clone = pred_kpts.detach().clone()[..., :2]
        stride_tensor_expand = (self._strides.unsqueeze(-1)).expand(-1, -1, 2, 2)
        right_selected_keypts = []
        right_pred_keypts_all = (pred_kpts_clone * stride_tensor_expand)
        for indexInBatch in range(num_batch):
            right_keypts_one = right_pred_keypts_all[indexInBatch][left_nms_topk_mask[indexInBatch]]
            pos_masks_one = left_fg_mask[indexInBatch][left_nms_topk_mask[indexInBatch]]
            right_selected_keypts.append(right_keypts_one[pos_masks_one])

        # # debug code. Replace right_selected_keypts with gt_selected_keypts.
        # gt_selected_keypts = []
        # for indexInBatch in range(num_batch):
        #     gt_keypointset_one = gt_keypoints[..., :2][stereo_objdet_targets["batch_idx"] == indexInBatch]
        #     gt_keypoints_one = gt_keypointset_one[left_target_gt_idx[indexInBatch][left_nms_topk_mask[indexInBatch]]]
        #     pos_masks_one = left_fg_mask[indexInBatch][left_nms_topk_mask[indexInBatch]]
        #     gt_selected_keypts.append(gt_keypoints_one[pos_masks_one])

        return loss_dict, list_sbboxes_pred_refined, pos_masks, right_selected_keypts

    @staticmethod
    def kpts_decode(anchor_points, pred_kpts):
        """Decodes predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y
    
    def compute_loss_yoloxformat(self, preds: Tuple[List, List, List], labels: Dict):
        """
        Args:
            preds:
            labels: dict containing the following keys.
                pos_masks: (B*100,)
                cls_targets: (sum(?), num_classes)
                bbox_targets: (sum(?), 6)
                indices_bbox_targets:
                batch_num_pos_per_img:
        """
        list_sbboxes_pred, list_refined_right_bboxes, list_right_scores_refine = preds
        pos_masks, cls_targets, bbox_targets, indices_bbox_targets, batch_num_pos_per_img = labels["pos_masks"], labels["cls_targets"], labels["bbox_targets"], labels["indices_bbox_targets"], labels["batch_num_pos_per_img"]
        loss_dict = {}
        num_batch = len(list_sbboxes_pred)
        list_sbboxes_pred_refined = []

        rbboxes_targets = torch.cat([
            bbox_targets[:, 4].unsqueeze(-1),
            bbox_targets[:, 1].unsqueeze(-1),
            bbox_targets[:, 5].unsqueeze(-1),
            bbox_targets[:, 3].unsqueeze(-1)
        ], dim=1)
        right_bboxes = torch.concat(list_refined_right_bboxes, dim=0)
        right_scores = torch.concat(list_right_scores_refine, dim=0)
        bboxes = torch.concat(list_sbboxes_pred, dim=0)  # shape [1, 100, 6]. (tl_x, tl_y, br_x, br_y, tl_x_r, br_x_r) format bbox, all in global scale.

        batch_size, num_priors, num_grids = right_bboxes.shape[:3]
        kernel_size = self._config['right_roi_feat_size']
        rbboxes_refined = right_bboxes.view(-1, num_grids, 4)[pos_masks]
        num_positive = rbboxes_refined.shape[0]
        ious, indicies_best_right = self.batch_iou_calculator_simple(rbboxes_refined, rbboxes_targets.unsqueeze(1))

        rselect_mask, rbboxes_refined_selected, rbboxes_targets_selected = self.batch_assigner(
            rbboxes_refined,
            ious,
            rbboxes_targets,
            self._config['r_iou_threshold'],
            self._config['candidates_k']
        )
        num_pos_timesk = torch.sum(rselect_mask.to(torch.float))
        num_total_samples_timesk = max(num_pos_timesk, 1.0)
        
        loss_rbbox = self.loss_rbbox(rbboxes_refined_selected, rbboxes_targets_selected) / num_total_samples_timesk

        # right scores
        rbboxes_scores = right_scores.view(-1, num_grids, 1)[pos_masks].sigmoid()
        ## dynamic targets
        rbboxes_scores_targets = torch.zeros_like(rbboxes_scores)
        rbboxes_scores_targets[rselect_mask] = 1
        
        loss_rscore = self.loss_rscore(rbboxes_scores, rbboxes_scores_targets.view(num_positive, -1, 1)) / num_total_samples_timesk

        # substitute right bboxes in sbboxes for visualization.
        bboxes = bboxes.view(-1, 6)
        selected_sbboxes = bboxes[pos_masks]
        indices_highest_score = torch.argmax(rbboxes_scores.squeeze(-1), dim=1)
        rbboxes_highest_score = torch.gather(rbboxes_refined, 1, indices_highest_score.view(num_positive, 1, 1).expand(-1, -1, 4)).squeeze(1)            
        selected_sbboxes[:, 4] = rbboxes_highest_score[:, 0]
        selected_sbboxes[:, 5] = rbboxes_highest_score[:, 2]
        bboxes[pos_masks] = selected_sbboxes
        bboxes = bboxes.view(batch_size, num_priors, 6)
        list_sbboxes_pred_refined.append(bboxes)
        # print("----- time sub sub2 loss stereo: {}".format(time.time() - starttime))
        
        loss_dict["loss_rbbox"] = loss_rbbox
        loss_dict["loss_rscore"] = loss_rscore

        return loss_dict, list_sbboxes_pred_refined

    def compute_loss_detrformat(self, preds: Tuple[List, List, List], labels: List[Dict]):
        list_sbboxes_pred, list_refined_right_bboxes, list_right_scores_refine = preds
        loss_dict = {}
        for indexInBatch in range(len(labels)):
            if list_sbboxes_pred[indexInBatch] is None:
                continue

            num_grids = list_refined_right_bboxes[indexInBatch].shape[2]
            rbboxes_targets = torch.cat([
                labels[indexInBatch]["bboxes"][:, 4].unsqueeze(-1),
                labels[indexInBatch]["bboxes"][:, 1].unsqueeze(-1),
                labels[indexInBatch]["bboxes"][:, 5].unsqueeze(-1),
                labels[indexInBatch]["bboxes"][:, 3].unsqueeze(-1)
            ], dim=1)

            rbboxes_refined = list_refined_right_bboxes[indexInBatch].view(-1, num_grids, 4)
            ious, indicies_best_right = self.batch_iou_calculator_simple(rbboxes_refined, rbboxes_targets.unsqueeze(1))

            # right bbox loss
            rselect_mask, rbboxes_refined_selected, rbboxes_targets_selected = self.batch_assigner(
                rbboxes_refined,
                ious,
                rbboxes_targets,
                self._config['r_iou_threshold'],
                self._config['candidates_k']
            )
            num_pos_timesk = torch.sum(rselect_mask.to(torch.float))
            num_total_samples_timesk = max(num_pos_timesk, 1.0)
            loss_rbbox = self.loss_rbbox(rbboxes_refined_selected, rbboxes_targets_selected) / num_total_samples_timesk            

            # right scores loss
            rbboxes_scores = list_right_scores_refine[indexInBatch].view(-1, num_grids, 1).sigmoid()
            ## dynamic targets
            rbboxes_scores_targets = torch.zeros_like(rbboxes_scores)
            rbboxes_scores_targets[rselect_mask] = 1
            loss_rscore = self.loss_rscore(rbboxes_scores, rbboxes_scores_targets) / num_total_samples_timesk

            if "loss_rbbox" in loss_dict:
                loss_dict["loss_rbbox"] += loss_rbbox
            else:
                loss_dict["loss_rbbox"] = loss_rbbox
            
            if "loss_rscore" in loss_dict:
                loss_dict["loss_rscore"] += loss_rscore
            else:
                loss_dict["loss_rscore"] = loss_rscore

        return loss_dict

    @torch.no_grad()
    def batch_iou_calculator_simple(self, bboxes_preds: Tensor, bboxes_ref: Tensor):
        """
        compute IoU between each pred at each batch and the only ref bbox at each batch.
        Args:
            bboxes_preds: [B, num_grids, 4]. box format (tl_x, tl_y, br_x, br_y).
            bboxes_ref: [B, 1, 4].

        Returns: 
            ious: [B,  num_grids]
            indicies_best_right: [B]
        """
        assert bboxes_ref.shape[1] == 1, "bboxes_ref should have only one at each batch."
        num_grids = bboxes_preds.shape[1]
        bboxes_ref = bboxes_ref.expand(-1, num_grids, -1)
        intersectionBox_tl_x = torch.max(torch.stack((bboxes_preds[..., 0], bboxes_ref[..., 0]), dim=-1), dim=-1)[0]
        intersectionBox_br_x = torch.min(torch.stack((bboxes_preds[..., 2], bboxes_ref[..., 2]), dim=-1), dim=-1)[0]
        
        ws = torch.clamp(intersectionBox_br_x - intersectionBox_tl_x, min=0, max=None)
        intersectionBox_tl_y = torch.max(torch.stack((bboxes_preds[..., 1], bboxes_ref[..., 1]), dim=-1), dim=-1)[0]
        intersectionBox_br_y = torch.min(torch.stack((bboxes_preds[..., 3], bboxes_ref[..., 3]), dim=-1), dim=-1)[0]
        hs = torch.clamp(intersectionBox_br_y - intersectionBox_tl_y, min=0, max=None)
        intersectionAreas = ws * hs  # [B, num_grids]
        unionAreas = (bboxes_ref[..., 2] - bboxes_ref[..., 0]) * (bboxes_ref[..., 3] - bboxes_ref[..., 1]) + (bboxes_preds[..., 2] - bboxes_preds[..., 0]) * (bboxes_preds[..., 3] - bboxes_preds[..., 1])
        ious = intersectionAreas / unionAreas
        return ious, torch.max(ious, dim=-1)[1]

    def batch_assigner(self, bboxes_preds: Tensor, iou_scores: Tensor, batch_gt_bboxes: Tensor, iou_thres: float, candidates_k: int):
        """
        for each gt, there are N candidates. Find the best candidates based on IoU_thres and candidates_k.
        If candidates within IoU_thres are less than candidates_k, 0 pad them.

        Args:
            bboxes_preds: shape (B, N, 4)
            iou_scores: shape (B, N, 1)
            batch_gt_bboxes: shape (B, 4)

        Returns:
            rselect_mask: shape (B, N), boolean mask.
            bboxes_preds_selected: shape (B, k, 4). Note some of the k elements can be just 0s.
            bboxes_targets_selected: shape (B, k, 4). Note some of the k elements can be just 0s.
        """
        bboxes_targets_selected = batch_gt_bboxes.view(-1, 1, 4).repeat(1, candidates_k, 1)
        batch_size, num_grids = bboxes_preds.shape[:2]

        with torch.no_grad():
            candidates_mask = iou_scores.squeeze(-1) > iou_thres;
            iou_scores_masked = iou_scores.squeeze(-1).clone().masked_fill(~candidates_mask, float('-inf'))
            topk_scores, topk_indices = torch.topk(iou_scores_masked, candidates_k, dim=1)

            valid_mask = topk_scores > float('-inf')
            valid_mask[:, 0] = True  # Note: make sure at least one candidate for each gt.
            topk_indices = topk_indices.masked_fill(~valid_mask, num_grids)  # shape (B, k)
        
        pseudo_bboxes = torch.zeros(batch_size, 1, 4, dtype=bboxes_preds.dtype, device=bboxes_preds.device)
        bboxes_preds_padded = torch.cat([bboxes_preds, pseudo_bboxes], dim=1)
        bboxes_preds_selected = torch.gather(bboxes_preds_padded, 1, topk_indices.unsqueeze(-1).expand(-1, -1, 4))
        bboxes_targets_selected = bboxes_targets_selected.masked_fill(~valid_mask.unsqueeze(-1).expand(-1, -1, 4), 0)

        # make sure at least one candidate for each gt
        indices_best_ious = torch.argmax(iou_scores, dim=1)
        candidates_mask[torch.arange(0, batch_size, device=candidates_mask.device), indices_best_ious] = True

        return candidates_mask, bboxes_preds_selected, bboxes_targets_selected


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

        self.bbox_roi_extractor_forfeature = MODELS.build(
            {
                'type': 'SingleRoIExtractor',
                'roi_layer': {
                    'type': 'RoIAlign',
                    'output_size': self._config.PARAMS['feat_size'],
                    'sampling_ratio': 0
                },
                'out_channels': self._config.PARAMS['in_channels'],
                'featmap_strides': [1]
            }
        )

        if not is_freeze:
            self.loss_featuremap = torch.nn.SmoothL1Loss(reduction='mean')

    @property
    def is_freeze(self):
        return self._config["is_freeze"]

    @property
    def input_shape(self):
        return [(1, 10, 480, 672), (1, 10, 480, 672)]

    def _init_layers(self) -> None:
        self.featmap_predictor = self._build_featmap_convs(
            in_channels=self._config.PARAMS['in_channels'],
            feat_channels=self._config.PARAMS['feat_channels'],
            num_classes=self._config.PARAMS["num_classes"],
        )

    def _init_weights(self):
        for m in self.featmap_predictor.modules():
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
            roi_feats = self.bbox_roi_extractor_forfeature([img_feats[indexInBatch].unsqueeze(0)], bnum_rois)  # Note: output shape is (b*100, 128, 14, 14)
            output_feat = self.featmap_predictor(roi_feats)  # Note: (b*100, num_of_class, 28, 28)
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
        img_feats: Tensor,
        bbox_preds: List[Tensor],
        class_preds: List[Tensor],
        feature_id: str,
        labels=None,
        **kwargs
    ):
        preds = self.predict(img_feats, bbox_preds, feature_id)
        losses = None
        artifacts = None
        if labels is not None and not self.is_freeze:
            losses, list_featmap_preds_inclass, list_mask_targets = self.compute_loss([preds, bbox_preds, class_preds], labels, "loss_" + feature_id)
            artifacts = (list_featmap_preds_inclass, list_mask_targets)
        return preds, losses, artifacts
    
    def compute_loss(
        self,
        preds: List[Tensor],
        labels: List[Tensor],
        lossKey: str
    ):
        list_featmap_preds, bbox_preds, class_preds = preds
        # list_featmap_preds[0] shape: (B, ?, num_classes, ker_h, ker_w)
        # bbox_preds[0] shape: (?, 4)
        # class_preds[0] shape: (?,)
        loss_dict = {}
        list_featmap_preds_inclass, list_mask_targets = [], []
        for indexInBatch in range(len(bbox_preds)):
            class_selection = class_preds[indexInBatch]
            featmap_size = list_featmap_preds[indexInBatch].shape[-1]
            num_targets = list_featmap_preds[indexInBatch].shape[1]
            imageHeight, imageWidth = labels[indexInBatch].shape[-2:]
            with torch.no_grad():
                pos_bboxes_pred_oneimg = bbox_preds[indexInBatch].detach()
                enlarge_factor = self._config.PARAMS["enlarge_roi_factor"]
                rois_w = (pos_bboxes_pred_oneimg[:, 2] - pos_bboxes_pred_oneimg[:, 0]) * enlarge_factor
                rois_h = (pos_bboxes_pred_oneimg[:, 3] - pos_bboxes_pred_oneimg[:, 1]) * enlarge_factor
                rois_centerX = (pos_bboxes_pred_oneimg[:, 2] + pos_bboxes_pred_oneimg[:, 0]) / 2
                rois_centerY = (pos_bboxes_pred_oneimg[:, 3] + pos_bboxes_pred_oneimg[:, 1]) / 2
                pos_bboxes_pred_oneimg[:, 0] = rois_centerX - rois_w / 2
                pos_bboxes_pred_oneimg[:, 1] = rois_centerY - rois_h / 2
                pos_bboxes_pred_oneimg[:, 2] = rois_centerX + rois_w / 2
                pos_bboxes_pred_oneimg[:, 3] = rois_centerY + rois_h / 2
            featmap_masks_oneimg = labels[indexInBatch]

            # mask_targets shape: (?, ker_h, ker_w)
            mask_targets = mask_target(
                [pos_bboxes_pred_oneimg],
                [torch.arange(0, num_targets)],
                [BitmapMasks(featmap_masks_oneimg.cpu().numpy(), height=imageHeight, width=imageWidth)],
                Config({"mask_size": featmap_size, "soft_mask_target": True})
            )

            featmap_preds = list_featmap_preds[indexInBatch].view(-1, self._config.PARAMS["num_classes"], featmap_size, featmap_size)
            featmap_preds_inclass = featmap_preds[torch.arange(0, featmap_preds.shape[0]), class_selection.view(-1).to(torch.int)]
            list_featmap_preds_inclass.append(featmap_preds_inclass)
            list_mask_targets.append(mask_targets)

        if len(list_featmap_preds_inclass) > 0:
            loss_dict[lossKey] = self.loss_featuremap(torch.cat(list_featmap_preds_inclass, dim=0), torch.cat(list_mask_targets, dim=0))  # Note: times 100 to increase loss magnitude
        
        return loss_dict, list_featmap_preds_inclass, list_mask_targets
    
    @staticmethod
    def ComputeCostProfile(model):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        img_feats = torch.randn((1, 10, 480, 672)).to(device)
        bbox_preds = [torch.randn((10, 4)).to(device)]
        bbox_preds[0][:, [0, 2]] *= 672
        bbox_preds[0][:, [1, 3]] *= 480
        class_preds = [torch.randn((10,)).to(device)]
        model = model.to(device)
        flops, numParams = profile(model, inputs=(img_feats, bbox_preds, class_preds, "profile"), verbose=False)
        return flops, numParams


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
