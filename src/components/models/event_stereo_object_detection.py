import torch.nn as nn
import torch
from torch import Tensor
import numpy
from typing import List, Dict, Tuple, Optional, Sequence, Union
from thop import profile
import cv2
import time
import math
import torch.nn.functional as F
from dataclasses import dataclass

from mmdet.registry import MODELS, TASK_UTILS
from mmengine.config import Config
from mmengine.model import initialize
from mmcv.cnn import ConvModule
from mmcv.ops import batched_nms
from mmdet.structures.mask import mask_target, BitmapMasks
from mmdet.models import CSPDarknet, YOLOXPAFPN

from .concentration import ConcentrationNet
from .stereo_matching import StereoMatchingNetwork
from .yolo_pose_utils import xyxy2xywh, initialize_weights
from .objectdetection import StereoEventDetectionHead
from .utils.objdet_utils import compute_ious_pertarget
from .rpn_base import RPNBaseClass
from .utils.misc import images_to_levels

from . import losses
from .losses import varifocal_loss
from .utils.misc import freeze_module_grads, multi_apply, convert_tensor_to_numpy
from ..methods.visz_utils import RenderImageWithBboxes, RenderImageWithBboxesAndKeypts


def EncodeKeypts(
    bbox_priors: Tensor,
    keypts_gt: Tensor,
    means: Optional[Sequence] = [0.0, 0.0],
    stds: Optional[Sequence] = [0.1, 0.1]
) -> Tensor:
    """
    Encode keypoints to the format of [delta_keypt0_x, delta_keypt0_y], wrt. bbox_priors.
    Args:
        bbox_priors: [N, 4] shape.
        keypts_gt: [N, 2] shape.

    Returns:
        encoded_right_keypts: [N, 2] shape.
    """
    px = (bbox_priors[..., 0] + bbox_priors[..., 2]) * 0.5
    py = (bbox_priors[..., 1] + bbox_priors[..., 3]) * 0.5
    pw = bbox_priors[..., 2] - bbox_priors[..., 0]
    ph = bbox_priors[..., 3] - bbox_priors[..., 1]

    dx = (keypts_gt[..., 0] - px) / pw
    dy = (keypts_gt[..., 1] - py) / ph
    deltas = torch.stack([dx, dy], dim=-1)
    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)
    return deltas

def DecodeKeypts(
    bbox_priors: Tensor,
    keypts_pred: Tensor,
    means: Optional[Sequence] = [0.0, 0.0],
    stds: Optional[Sequence] = [0.1, 0.1]
) -> Tensor:
    """
    Decode keypoints given bbox_priors and keypts_pred (deltas).
    """
    deltas = keypts_pred.view(-1, 2)

    means = deltas.new_tensor(means).view(1, -1)
    stds = deltas.new_tensor(stds).view(1, -1)
    denorm_deltas = deltas * stds + means

    dxy = denorm_deltas[:, :2]
    pxy = ((bbox_priors[:, :2] + bbox_priors[:, 2:]) * 0.5)
    pwh = (bbox_priors[:, 2:] - bbox_priors[:, :2])

    dxy_wh = pwh * dxy
    decoded_keypts = pxy + dxy_wh
    return decoded_keypts


@dataclass
class RPNHypothesesGroup:
    bboxes: Optional[Tensor] = None
    scores: Optional[Tensor] = None
    level_ids: Optional[Tensor] = None
    target_ids: Optional[Tensor] = None

    def __getitem__(self, maskOrIndicesOrString: Tensor) -> Union['RPNHypothesesGroup', dict]:
        """
        Args:
            maskOrIndicesOrString: mask tensor or indices tensor of existing hypotheses.

        Returns:
            - attr: behave like a dict if key is a string same as an attr variable.
            - RPNHypothesesGroup with all member variables filtered by mask or indices tensor.
        """
        if isinstance(maskOrIndicesOrString, str):
            # behave like a dict if key is a string
            return getattr(self, maskOrIndicesOrString)
        else:
            return RPNHypothesesGroup(
                bboxes=self.bboxes[maskOrIndicesOrString],
                scores=self.scores[maskOrIndicesOrString],
                level_ids=self.level_ids[maskOrIndicesOrString],
                target_ids=self.target_ids[maskOrIndicesOrString]
            )

    def __add__(self, other: 'RPNHypothesesGroup') -> 'RPNHypothesesGroup':
        return RPNHypothesesGroup(
            bboxes=torch.cat([self.bboxes, other.bboxes], dim=0) if self.bboxes is not None and other.bboxes is not None else None,
            scores=torch.cat([self.scores, other.scores], dim=0) if self.scores is not None and other.scores is not None else None,
            level_ids=torch.cat([self.level_ids, other.level_ids], dim=0) if self.level_ids is not None and other.level_ids is not None else None,
            target_ids=torch.cat([self.target_ids, other.target_ids], dim=0) if self.target_ids is not None and other.target_ids is not None else None,
        )
    
    def zero(self) -> 'RPNHypothesesGroup':
        device = self.bboxes.device if self.bboxes is not None else 'cpu'
        return RPNHypothesesGroup(
            bboxes=torch.zeros((0, 4), dtype=torch.float32, device=device),
            scores=torch.zeros((0,), dtype=torch.float32, device=device),
            level_ids=torch.zeros((0,), dtype=torch.long, device=device),
            target_ids=torch.zeros((0,), dtype=torch.long, device=device)
        )

    def numpy(self) -> 'RPNHypothesesGroup':
        return RPNHypothesesGroup(
            bboxes=self.bboxes.detach().cpu().numpy() if self.bboxes is not None else None,
            scores=self.scores.detach().cpu().numpy() if self.scores is not None else None,
            level_ids=self.level_ids.detach().cpu().numpy() if self.level_ids is not None else None,
            target_ids=self.target_ids.detach().cpu().numpy() if self.target_ids is not None else None,
        )
    
    def get_dict(self) -> dict:
        dictData = {
            "bboxes": self.bboxes,
            "scores": self.scores,
            "target_ids": self.target_ids
        }
        if self.level_ids is not None:
            dictData["level_ids"] = self.level_ids
        return dictData


def prune_hypotheses_equally_for_targets(hypotheses: RPNHypothesesGroup, max_hypotheses_per_image: int) -> RPNHypothesesGroup:
    unique_target_ids = torch.unique(hypotheses.target_ids)
    max_hypotheses_per_target = max_hypotheses_per_image // unique_target_ids.numel()
    pruned_hypotheses = []
    for tid in unique_target_ids:
        mask = hypotheses.target_ids == tid
        # sort by scores and keep the top max_hypotheses_per_target
        sorted_scores, sorted_indices = hypotheses.scores[mask].sort(descending=True)
        if mask.sum() > max_hypotheses_per_target:
            top_indices = sorted_indices[:max_hypotheses_per_target]
            mask = torch.zeros_like(hypotheses.target_ids, dtype=torch.bool)
            mask[torch.where(hypotheses.target_ids == tid)[0][top_indices]] = True
            pruned_hypotheses.append(hypotheses[mask])
        else:
            pruned_hypotheses.append(hypotheses[mask][sorted_indices])
    pruned_hypotheses = sum(pruned_hypotheses, hypotheses.zero())
    return pruned_hypotheses


class RPNWithTargetsHead(RPNBaseClass):
    def __init__(
        self,
        network_cfg: dict,
        **kwargs      
    ):
        super(RPNWithTargetsHead, self).__init__()
        
        self._config = network_cfg
        self.feat_channels = 256
        self.in_channels = 256
        self.num_classes = 1  # rpn only classifies foreground and background.
        self.cls_out_channels = self.num_classes
        anchors_ratios = [0.5, 1.0, 2.0]
        self.prior_generator = TASK_UTILS.build(
            {
                'type': 'AnchorGenerator',
                'scales': [8],
                'ratios': anchors_ratios,
                'strides': [4, 8, 16, 32]
            }
        )
        self.num_base_priors = len(anchors_ratios)
        self.bbox_coder = TASK_UTILS.build(
            {
                'type': 'DeltaXYWHBBoxCoder',
                'target_means': [0.0, 0.0, 0.0, 0.0],
                'target_stds': [1.0, 1.0, 1.0, 1.0]
            }
        )
        self.loss_cls = MODELS.build(
            {'type': 'CrossEntropyLoss', 'use_sigmoid': True, 'loss_weight': 1.0}
        )
        self.loss_bbox = MODELS.build(
            {'type': 'L1Loss', 'loss_weight': 1.0}
        )
        self.assigner = TASK_UTILS.build(
            {
                'type': 'MaxIoUAssigner',
                'pos_iou_thr': 0.7,
                'neg_iou_thr': 0.3,
                'min_pos_iou': 0.3,
                'match_low_quality': True,
                'ignore_iof_thr': -1
            }
        )
        self.sampler = TASK_UTILS.build(
            {
                'type': 'RandomSampler',
                'num': 256,
                'pos_fraction': 0.5,
                'neg_pos_ub': -1,
                'add_gt_as_proposals': False
            },
            default_args=dict(context=self)
        )
        self.init_layers()

    @property
    def config(self):
        return self._config

    def init_layers(self)->tuple:
        self.rpn_conv = nn.Conv2d(
            self.in_channels,
            self.feat_channels,
            3,
            padding=1
        )
        self.rpn_cls = nn.Conv2d(
            self.feat_channels,
            self.num_base_priors * self.cls_out_channels,
            1
        )
        # reg_dim = self.bbox_coder.encode_size
        reg_dim = 4
        self.rpn_reg = nn.Conv2d(
            self.feat_channels,
            self.num_base_priors * reg_dim,
            1
        )
    
    def predict(self, xs: list[Tensor]) -> tuple[List[Tensor], List[Tensor]]:
        """
        Args:
            xs: list of feature maps, each of shape (B, feat_channels, H//k, W//k).
        """
        listfeat_rpn_cls_score, listfeat_rpn_bbox_pred = multi_apply(self.predict_single, xs)
        return listfeat_rpn_cls_score, listfeat_rpn_bbox_pred

    def predict_single(self, x):
        """
        Args:
            x: feature map of one scale, shape (B, feat_channels, H//k, W//k)
        """
        x = self.rpn_conv(x)
        x = F.relu(x)
        rpn_cls_score = self.rpn_cls(x)
        rpn_bbox_pred = self.rpn_reg(x)
        return rpn_cls_score, rpn_bbox_pred
    
    def allocate_hypotheses_to_targets(
        self,
        listfeat_rpn_cls_score,
        listfeat_rpn_bbox_pred,
        batch_targets,
        imageHeight,
        imageWidth,
        min_iou_with_target,
    ):
        list_hypotheses = self._allocate_hypotheses_to_targets(
            listfeat_rpn_cls_score,
            listfeat_rpn_bbox_pred,
            batch_targets,
            imageHeight,
            imageWidth,
            min_iou_with_target=min_iou_with_target,
        )
        return list_hypotheses
    
    def _allocate_hypotheses_to_targets(
        self,
        listfeat_cls_score: list[Tensor],
        listfeat_bbox_pred: list[Tensor],
        targets: List[Tensor],
        imageHeight,
        imageWidth,
        nms_pred: int = 2000,
        max_hypotheses_per_img: int = 1000,
        min_iou_with_target: float = 0.0,
        min_bbox_size: float = 0.0
    ) -> List[RPNHypothesesGroup]:
        """
        Allocate K best hypotheses to each target.
        """
        num_images_in_batch = len(targets)
        num_levels = len(listfeat_cls_score)

        featmap_sizes = [listfeat_cls_score[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=listfeat_cls_score[0].dtype,
            device=listfeat_cls_score[0].device)

        reg_dim = self.bbox_coder.encode_size
        batch_results = []
        for indexInBatch in range(num_images_in_batch):
            mlvl_bbox_preds = []
            mlvl_valid_priors = []
            mlvl_scores = []
            level_ids = []  # need for nms in separating level
            cls_score_singleimage = [cls_score[indexInBatch] for cls_score in listfeat_cls_score]
            bbox_pred_singleimage = [bbox_pred[indexInBatch] for bbox_pred in listfeat_bbox_pred]
            for indexLevel, (cls_score_onelvl, bbox_pred_onelvl, priors) in enumerate(zip(cls_score_singleimage, bbox_pred_singleimage, mlvl_priors)):
                bbox_pred_onelvl = bbox_pred_onelvl.permute(1, 2, 0).reshape(-1, reg_dim)
                cls_score_onelvl = cls_score_onelvl.permute(1, 2, 0).reshape(-1, self.cls_out_channels)
                scores = cls_score_onelvl.sigmoid()

                scores = torch.squeeze(scores)
                if 0 < nms_pred < scores.shape[0]:
                    # sort is faster than topk
                    ranked_scores, rank_inds = scores.sort(descending=True)
                    topk_inds = rank_inds[:nms_pred]
                    scores = ranked_scores[:nms_pred]
                    bbox_pred_onelvl = bbox_pred_onelvl[topk_inds, :]
                    priors = priors[topk_inds]

                mlvl_bbox_preds.append(bbox_pred_onelvl)
                mlvl_valid_priors.append(priors)
                mlvl_scores.append(scores)

                # use level id to implement the separate level nms
                level_ids.append(
                    scores.new_full(
                        (scores.size(0),),
                        indexLevel,
                        dtype=torch.long
                    )
                )
            bbox_pred = torch.cat(mlvl_bbox_preds)  # [N, 4]
            priors = torch.cat(mlvl_valid_priors)  # [N, 4]
            # restore bboxes to image pixel coordinates
            bboxes_xyxy = self.bbox_coder.decode(priors, bbox_pred, max_shape=(imageHeight, imageWidth))
            num_targets = targets[indexInBatch].shape[0]

            foundNoResults = False
            if num_targets == 0:
                foundNoResults = True
            else:
                ious = compute_ious_pertarget(bboxes_xyxy, targets[indexInBatch])
                # remove targets with no ious larger than 0
                notarget_mask = ious.max(dim=1).values <= min_iou_with_target
                bboxes_xyxy = bboxes_xyxy[~notarget_mask]
                mlvl_scores = torch.cat(mlvl_scores)[~notarget_mask]
                level_ids = torch.cat(level_ids)[~notarget_mask]
                target_ids = ious.argmax(dim=1)[~notarget_mask]
                results = RPNHypothesesGroup(bboxes=bboxes_xyxy, scores=mlvl_scores, level_ids=level_ids, target_ids=target_ids)

                # filter small size bboxes
                w, h = results.bboxes[:, 2] - results.bboxes[:, 0], results.bboxes[:, 3] - results.bboxes[:, 1]
                valid_mask = (w > min_bbox_size) & (h > min_bbox_size)
                if not valid_mask.all():
                    results = results[valid_mask]

                # another round of nms to reduce number to max_hypotheses_per_img
                if results.bboxes.numel() > 0:
                    det_bboxes, keep_idxs = batched_nms(
                        results.bboxes,
                        results.scores,
                        results.level_ids * num_targets + results.target_ids,  # Note: hypotheses of differnet level and different target id will not merge.
                        {'type': 'nms', 'iou_threshold': 0.7}
                    )
                    results = results[keep_idxs]
                    results.scores = det_bboxes[:, -1]
                    results = prune_hypotheses_equally_for_targets(results, max_hypotheses_per_img)
                    del results.level_ids
                    results.bboxes = results.bboxes.detach().clone()  # in RPN, no need to use prediction results to compute losses.
                    results.scores = results.scores.detach().clone()
                else:
                    foundNoResults = True

            if foundNoResults:
                # To avoid some potential error
                results_ = RPNHypothesesGroup()
                results_.bboxes = bbox_pred.new_zeros(0)
                results_.scores = bbox_pred[:, 0].new_zeros(0)
                results = results_
            batch_results.append(results)
        return batch_results
    
    def compute_loss(
        self,
        cls_scores: List[Tensor],
        bbox_preds: List[Tensor],
        labels: Dict,
        img_metas: Dict
    ):
        """
        Args:
            cls_scores: Box scores for each scale level has shape (B, num_anchors * 1, H, W).
            bbox_preds: Box regression predictions for each scale level has shape (B, num_anchors * 4, H, W).
            labels: dict
                - stereo_objdet_targets: dict
                    - bboxes: (N_allbatch, 6)
                    - cls: (N_allbatch,)
                    - batch_idx: (N_allbatchh,).
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        batch_size = cls_scores[0].shape[0]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        # set cat_id of gt_labels to 0 in RPN
        cls_scores_gt = labels['stereo_objdet_targets']['cls']
        cls_scores_gt = torch.zeros_like(cls_scores_gt)
        device = cls_scores[0].device
        batch_img_metas = [{
            'img_shape': (img_metas["h_recti"], img_metas["w_recti"]),
            'pad_shape': (img_metas["h"], img_metas["w"]),
        }] * batch_size

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes,
            batch_img_metas,
            device=device
        )
        batch_labels = [{
                'bboxes': labels['stereo_objdet_targets']['bboxes'][labels['stereo_objdet_targets']['batch_idx'] == indexInBatch][:, [4, 1, 5, 3]],
                'labels': cls_scores_gt[labels['stereo_objdet_targets']['batch_idx'] == indexInBatch]
            } for indexInBatch in range(batch_size)
        ]
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            batch_labels,
            batch_img_metas
        )
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, avg_factor) = cls_reg_targets

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(torch.cat(anchor_list[i], dim=0))
        all_anchor_list = images_to_levels(concat_anchor_list, num_level_anchors)

        losses_cls, losses_bbox = multi_apply(
            self.compute_loss_single,
            cls_scores,
            bbox_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            avg_factor=avg_factor
        )
        return losses_cls, losses_bbox


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

        self._config["loss_cfg"] = loss_cfg

        # objdet tools
        backboneCfg = {
            "stem_in_channels": 10,
            "deepen_factor": 0.33,
            "widen_factor": 0.5,
            "out_indices": [1, 2, 3, 4],
            "use_depthwise": False,
            "spp_kernal_sizes": [5, 9, 13],
            "norm_cfg": {
              "type": 'BN',
              "momentum": 0.03,
              "eps": 0.001
            },
            "act_cfg": {"type": 'Swish'}
        }
        neckCfg = {
            "in_channels": [64, 128, 256, 512],
            "out_channels": self._config['channels_roi_extraction'],
            "num_csp_blocks": 1,
            "use_depthwise": False,
            "upsample_cfg": {
                "scale_factor": 2,
                "mode": "nearest"
            },
            "norm_cfg": {
                "type": "BN",
                "momentum": 0.03,
                "eps": 0.001
            },
            "act_cfg": {"type": "Swish"}
        }
        self.backbone = CSPDarknet(**backboneCfg)
        self.neck = YOLOXPAFPN(**neckCfg)
        self.rpn_head = RPNWithTargetsHead(self._config)

        self.bbox_roi_extractor = MODELS.build(
            {
                'type': 'SingleRoIExtractor',
                'roi_layer': {'type': 'RoIAlign', 'output_size': self._config['right_roi_feat_size'], 'sampling_ratio': 0},
                'out_channels': self._config['channels_roi_extraction'],
                'featmap_strides': [4, 8, 16, 32]
            }
        )

        self.bbox_coder = TASK_UTILS.build(
            {
                'type': 'DeltaXYWHBBoxCoder',
                'target_means': [0.0, 0.0, 0.0, 0.0],
                'target_stds': [0.1, 0.1, 0.2, 0.2]
            }
        )

        (
            self.shared_convs, # empy
            self.shared_fcs,  # two layers
            last_layer_dim  # 1024
        ) = self._add_conv_fc_branch(
            num_branch_convs=0,
            num_branch_fcs=2,
            in_channels=self._config['channels_roi_extraction'],
            conv_out_channels=256,
            num_shared_fcs=2,
            with_avg_pool=False,
            fc_out_channels=1024,
            roi_feat_area=self._config['right_roi_feat_size'] ** 2,
            is_shared=True
        )
        self.shared_out_channels = last_layer_dim

        # output fc layers
        box_dim = 4
        out_dim_reg = box_dim * self._config['num_classes']  # Note: classify again in the right side.
        reg_last_dim = last_layer_dim
        self.fc_reg = MODELS.build({'type': 'Linear', 'in_features': reg_last_dim, 'out_features': out_dim_reg})
        cls_last_dim = last_layer_dim  # 1024
        cls_channels = self._config['num_classes'] + 1
        self.fc_cls = MODELS.build({'type': 'Linear', 'in_features': cls_last_dim, 'out_features': cls_channels})
        keypts_dim = 3  # only one keypts for these objects for now.
        keypts_out_dim = keypts_dim * self._config['num_classes'] * self._config['max_num_keypoints']
        keypts_last_dim = last_layer_dim
        self.fc_keypts = MODELS.build({'type': 'Linear', 'in_features': keypts_last_dim, 'out_features': keypts_out_dim})

        self.relu = nn.ReLU(inplace=True)

        init_cfg = [
            dict(
                type='Xavier',
                distribution='uniform',
                override=[
                    dict(name='shared_fcs'),
                ]
            )
        ]
        initialize(self, init_cfg)
        
        # loss
        self.loss_rscore = varifocal_loss
        self.loss_rbbox = MODELS.build({'type': 'L1Loss', 'loss_weight': 1.0})
        self.loss_bce_pose = nn.BCEWithLogitsLoss()
        OKS_SIGMA = (
            numpy.array(self._config['OKS_SIGMA'][:self._config['max_num_keypoints']])  # Note: hard-coded values
            / 10.0
        )
        self.loss_keypoint = losses.KeypointLoss(oks_sigmas=OKS_SIGMA)

        # self._init_layers()
        # self._init_weights()

        self.logger = kwargs.get("logger", None)

    @property
    def is_freeze(self):
        return self._config["is_freeze"]
    
    @property
    def config(self):
        return self._config

    def _add_conv_fc_branch(
        self,
        num_branch_convs: int,
        num_branch_fcs: int,
        in_channels: int,
        conv_out_channels: int,
        num_shared_fcs: int,
        with_avg_pool: bool,
        fc_out_channels: int,
        roi_feat_area: int,
        is_shared: bool = False
    ) -> tuple:
        """Add shared or separable branch.

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg
                    )
                )
            last_layer_dim = conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared or num_shared_fcs == 0) and not with_avg_pool:
                last_layer_dim *= roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (last_layer_dim if i == 0 else fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, fc_out_channels)
                )
            last_layer_dim = fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    # def _init_layers(self):
    #     # self.backbone_net = torch.nn.Sequential(
    #     #     Conv(10, 16, 3, 2),
    #     #     C3k2(16, 32, 1, False, 0.25),
    #     #     Conv(32, 64, 3, 2),
    #     #     C3k2(64, 128, 1, False, 0.25),
    #     #     Conv(128, 128, 1, 1)
    #     # )
    #     # self.downsample_layer = Conv(128, 128, 3, 2)
    #     self.right_bbox_refiner = self._build_bbox_refiner_convs(
    #         self._config['in_channels'],
    #         self._config['feat_channels'],
    #         output_logits=4,
    #         norm_eps=self._config["norm_cfg"]["eps"],
    #         norm_momentum=self._config["norm_cfg"]["momentum"],
    #         act_type=self._config["act_cfg"]["type"]
    #     )
    #     self.right_bbox_refiner_scorer = self._build_bbox_refiner_convs(
    #         self._config['in_channels'],
    #         self._config['feat_channels'],
    #         output_logits=1,
    #         norm_eps=self._config["norm_cfg"]["eps"],
    #         norm_momentum=self._config["norm_cfg"]["momentum"],
    #         act_type=self._config["act_cfg"]["type"]
    #     )

    #     self.right_keypts_predictor = self._build_bbox_refiner_convs(
    #         self._config['in_channels'],
    #         self._config['feat_channels'],
    #         output_logits=self._config["max_num_keypoints"] * 3,
    #         norm_eps=self._config["norm_cfg"]["eps"],
    #         norm_momentum=self._config["norm_cfg"]["momentum"],
    #         act_type=self._config["act_cfg"]["type"]
    #     )
    #     self.right_keypts_predictor_scorer = self._build_bbox_refiner_convs(
    #         self._config['in_channels'],
    #         self._config['feat_channels'],
    #         output_logits=1,
    #         norm_eps=self._config["norm_cfg"]["eps"],
    #         norm_momentum=self._config["norm_cfg"]["momentum"],
    #         act_type=self._config["act_cfg"]["type"]
    #     )

    # def _init_weights(self):
    #     """
    #     all conv2d need weights initialization
    #     """
    #     for subnet in [self.right_bbox_refiner, self.right_keypts_predictor]:
    #         for module in subnet.modules():
    #             if isinstance(module, (nn.Conv2d, nn.Linear)):
    #                 nn.init.kaiming_uniform_(
    #                     module.weight,
    #                     a=math.sqrt(5),
    #                     mode="fan_in",
    #                     nonlinearity="leaky_relu"
    #                 )
    #     # initialize_weights(self.backbone_net)
    #     # initialize_weights(self.downsample_layer)

    # @staticmethod
    # def _build_bbox_refiner_convs(
    #     in_channels: int,
    #     feat_channels: int,
    #     output_logits: int,
    #     kernel_size: int = 3,
    #     stride: int = 1,
    #     padding: int = 1,
    #     norm_eps: float = 1e-5,
    #     norm_momentum: float = 0.1,
    #     act_type: str = "ReLU"
    # ) -> nn.Sequential:
    #     """
    #     For left, output logits are [delta_x, delta_y, w, h], w and h are relative values to bbox size;
    #     For right, output two logits are [delta_x, w].
    #     """
    #     bbox_refiner = nn.Sequential(
    #         nn.Conv2d(in_channels, feat_channels, kernel_size, stride=stride, padding=padding),
    #         nn.BatchNorm2d(feat_channels, eps=norm_eps, momentum=norm_momentum),
    #         getattr(nn, act_type)(),
    #         nn.Conv2d(feat_channels, feat_channels, kernel_size, stride=stride, padding=padding),
    #         nn.BatchNorm2d(feat_channels, eps=norm_eps, momentum=norm_momentum),
    #         getattr(nn, act_type)(),
    #         nn.Conv2d(feat_channels, output_logits, 1)
    #     )
    #     return bbox_refiner

    @staticmethod
    def ComputeCostProfile(model):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        h, w = 480, 672
        event_voxel = torch.randn(4, 10, 480, 672).to(device)
        bboxes = [torch.randn(120, 4).to(device)]
        disp_prior = torch.randn(4, h, w).to(device)
        batch_img_metas = {"h": h, "w": w}
        model = model.to(device)
        flops, numParams = profile(model, inputs=(event_voxel,  bboxes, disp_prior), verbose=False)
        return flops, numParams
    
    # def _right_bbox_decode(self, sbboxes_pred: Tensor, right_boxes_refine: Tensor) -> Tensor:
    #     """
    #     Decode right bbox refine result [B, 100, 4, ker_h, ker_w] whose '4' dimension is (delta_x, delta_y, w_factor, h_factor) to
    #     same shape whose '4' dimension is (tl_x_r, tl_y_r, br_x_r, br_y_r).
    #     Args:
    #         sbboxes_pred:  [B, 100, 6]
    #         right_boxes_refine: [B, 100, 4, ker_h, ker_w]
        
    #     Return:
    #         decoded_right_bboxes: [B, 100, ker_h, ker_w, 4] whose '4' dimension is (tl_x_r, tl_y_r, br_x_r, br_y_r)
    #     """
    #     batch_size, num_samples = sbboxes_pred.shape[:2]
    #     ker_h, ker_w = right_boxes_refine.shape[-2:]
    #     sbboxes_pred = sbboxes_pred.view(-1, 6)
    #     right_boxes_refine = right_boxes_refine.view(-1, 4, ker_h, ker_w)
        
    #     strides_x = (sbboxes_pred[:, 5] - sbboxes_pred[:, 4]) / ker_w
    #     strides_y = (sbboxes_pred[:, 3] - sbboxes_pred[:, 1]) / ker_h
    #     strides_x = strides_x.view(-1, 1, 1).expand(-1, ker_h, ker_w).unsqueeze(1)
    #     strides_y = strides_y.view(-1, 1, 1).expand(-1, ker_h, ker_w).unsqueeze(1)
    #     grid_x = torch.arange(0, ker_w, device=sbboxes_pred.device, dtype=sbboxes_pred.dtype).view(1, 1, -1).expand(batch_size * num_samples, ker_h, -1) * strides_x.squeeze(1)
    #     grid_x += sbboxes_pred[:, 4].clone().view(-1, 1, 1).expand(-1, ker_h, ker_w)
    #     grid_x = grid_x.unsqueeze(1)
    #     grid_y = torch.arange(0, ker_h, device=sbboxes_pred.device, dtype=sbboxes_pred.dtype).view(1, -1, 1).expand(batch_size * num_samples, -1, ker_w) * strides_y.squeeze(1)
    #     grid_y += sbboxes_pred[:, 1].clone().view(-1, 1, 1).expand(-1, ker_h, ker_w)
    #     grid_y = grid_y.unsqueeze(1)

    #     strides = torch.cat([strides_x, strides_y], dim=1)  # [B*100, 2, ker_h, ker_w]
    #     grids = torch.cat([grid_x, grid_y], dim=1)  # [B*100, 2, ker_h, ker_w]
    #     xys = right_boxes_refine[:, :2, :, :] * strides + grids
    #     whs = right_boxes_refine[:, 2:, :, :].exp() * strides

    #     tl_x = (xys[:, 0, :, :] - whs[:, 0, :, :] / 2).unsqueeze(-1)
    #     tl_y = (xys[:, 1, :, :] - whs[:, 1, :, :] / 2).unsqueeze(-1)
    #     br_x = (xys[:, 0, :, :] + whs[:, 0, :, :] / 2).unsqueeze(-1)
    #     br_y = (xys[:, 1, :, :] + whs[:, 1, :, :] / 2).unsqueeze(-1)

    #     decoded_right_bboxes = torch.cat([tl_x, tl_y, br_x, br_y], dim=-1)
    #     return decoded_right_bboxes.view(batch_size, num_samples, ker_h * ker_w, 4)
    
    # def _right_keypts_decode(self, sbboxes_pred: Tensor, right_keypts_pred: Tensor) -> Tensor:
    #     """
    #     Decode right keypts prediction [B, 100, 3*num_keypts, ker_h, ker_w] whose '6' dimension is (delta_keypt0_x, delta_keypt0_y, visibility, delta_keypt1_x, delta_keypt1_y, visibility) to same shape whose '6'
    #     dimension is (keypt0_x, keypt0_y, visibility, keypt1_x, keypt1_y, visibility).
    #     Args:
    #         sbboxes_pred: [B, 100, 6]
    #         right_keypts_pred: [B, 100, 3*num_keypts, ker_h, ker_w]

    #     Return:
    #         decoded_right_keypts: [B, 100, ker_h, ker_w, 3*num_keypts] whose '3*num_keypts' dimension is (keypt0_x, keypt0_y, visibility, keypt1_x, keypt1_y, visibility, ...).
    #     """
    #     max_num_keypoints = self._config["max_num_keypoints"]

    #     # ----------------- this part is same as _right_bbox_decode. TODO: reuse this part more -----------------
    #     batch_size, num_samples = sbboxes_pred.shape[:2]
    #     ker_h, ker_w = right_keypts_pred.shape[-2:]
    #     sbboxes_pred = sbboxes_pred.view(-1, 6)
    #     right_keypts_pred = right_keypts_pred.view(-1, 3*max_num_keypoints, ker_h, ker_w)

    #     strides_x = (sbboxes_pred[:, 5] - sbboxes_pred[:, 4]) / ker_w
    #     strides_y = (sbboxes_pred[:, 3] - sbboxes_pred[:, 1]) / ker_h
    #     strides_x = strides_x.view(-1, 1, 1).expand(-1, ker_h, ker_w).unsqueeze(1)
    #     strides_y = strides_y.view(-1, 1, 1).expand(-1, ker_h, ker_w).unsqueeze(1)
    #     grid_x = torch.arange(0, ker_w, device=sbboxes_pred.device, dtype=sbboxes_pred.dtype).view(1, 1, -1).expand(batch_size * num_samples, ker_h, -1) * strides_x.squeeze(1)
    #     grid_x += sbboxes_pred[:, 4].clone().view(-1, 1, 1).expand(-1, ker_h, ker_w)
    #     grid_x = grid_x.unsqueeze(1)
    #     grid_y = torch.arange(0, ker_h, device=sbboxes_pred.device, dtype=sbboxes_pred.dtype).view(1, -1, 1).expand(batch_size * num_samples, -1, ker_w) * strides_y.squeeze(1)
    #     grid_y += sbboxes_pred[:, 1].clone().view(-1, 1, 1).expand(-1, ker_h, ker_w)
    #     grid_y = grid_y.unsqueeze(1)

    #     strides = torch.cat([strides_x, strides_y], dim=1)  # [B*100, 2, ker_h, ker_w]
    #     grids = torch.cat([grid_x, grid_y], dim=1)  # [B*100, 2, ker_h, ker_w]
    #     # ----------------- this part is same as _right_bbox_decode -----------------

    #     keypts_list = []
    #     for iKeypt in range(max_num_keypoints):
    #         keypt_xys = right_keypts_pred[:, (3*iKeypt):(2+3*iKeypt), :, :] * strides + grids
    #         keypts_list.append(keypt_xys)
    #         keypts_list.append(right_keypts_pred[:, 2+3*iKeypt, :, :].unsqueeze(1))  # visibility
    #     decoded_right_keypts = torch.cat(keypts_list, dim=1).permute(0, 2, 3, 1)
    #     return decoded_right_keypts.view(batch_size, num_samples, ker_h * ker_w, 3 * max_num_keypoints)

    # def extract_pyramid_featuremaps(self, input_feat: Tensor) -> List[Tensor]:
    #     """
    #     Extract feature maps from backbone network.
    #     Args:
    #         input_feat: shape [B, 10, h, w]
        
    #     Returns:
    #         pyramid_feats: [[B,128, h/4, w/4], [B,128, h/8, w/8], [B,128, h/16, w/16]]
    #     """
    #     pyramid_feats = []
    #     pyramid_feats.append(self.backbone_net(input_feat))
    #     pyramid_feats.append(self.downsample_layer(pyramid_feats[-1]))
    #     pyramid_feats.append(self.downsample_layer(pyramid_feats[-1]))
    #     return pyramid_feats

    def warp_bboxes(
        self,
        left_bboxes: List[Tensor],
        disp_prior: Tensor,
        imageHeight: int,
        imageWidth: int
    ):
        """
        warp left bboxes by disparity prior.
        """
        warped_bboxes = []
        left_bboxes = [bboxes.detach().clone() for bboxes in left_bboxes]  # avoid inplace operation
        for indexInBatch, bboxes_oneimage in enumerate(left_bboxes):
            xindi = ((bboxes_oneimage[..., 0] + bboxes_oneimage[..., 2]) / 2).to(torch.int).clamp(0, imageWidth - 1).squeeze()
            yindi = ((bboxes_oneimage[..., 1] + bboxes_oneimage[..., 3]) / 2).to(torch.int).clamp(0, imageHeight - 1).squeeze()
            bbox_disps = disp_prior[indexInBatch][yindi, xindi]
            if bbox_disps.dim() == 0:
                bbox_disps = bbox_disps.unsqueeze(0)
            bboxes_oneimage[..., [0, 2]] -= bbox_disps.unsqueeze(-1).expand(-1, 2)
            warped_bboxes.append(bboxes_oneimage)
        return warped_bboxes

    def predict(
        self,
        right_event_voxel: Tensor,
        left_bboxes: List[Tensor],
        disp_prior: Tensor,
        batch_hypotheses: Optional[List[RPNHypothesesGroup]] = None
    ) -> Tuple[List[Optional[Tensor]], List[Optional[Tensor]], List[Optional[Tensor]]]:
        """
        Args:
            right_event_voxel: shape is [B, 10, h, w]
            bboxes_pred: list of B tensors of shape [?, 4]. [tl_x, tl_y, br_x, br_y] format bbox, all in global scale.
            disp_prior: [B, h, w] shape.

        Returns:
            sbboxes_priors: list of shape [N, 8]. format [tl_x, tl_y, br_x, br_y, tl_x_r, br_x_r] rough stereo bbox
            list_corresponding_leftdet_ids: list of shape [N,], target ids for each prior.
            refined_right_bboxes: shape [B, N, ker_h * ker_w, 4]. Corresponding refined right bboxes.
            right_scores_refine: shape [B, N, ker_h * ker_w, 1]. Corresponding scores.
        """
        list_sbboxes_priors, list_corresponding_leftdet_ids, list_refined_right_bboxes, list_refined_right_scores, list_predicted_right_keypts = [], [], [], [], []
        right_feats = self.backbone(right_event_voxel)
        right_feats = self.neck(right_feats)
        if batch_hypotheses is None:
            device = right_event_voxel.device
            imageHeight, imageWidth = right_event_voxel.shape[-2:]
            warped_left_bboxes = self.warp_bboxes(left_bboxes, disp_prior, imageHeight=imageHeight, imageWidth=imageWidth)
            rpn_cls_scores, rpn_bbox_preds = self.rpn_head.predict(right_feats)
            batch_hypotheses = self.rpn_head.allocate_hypotheses_to_targets(
                rpn_cls_scores,
                rpn_bbox_preds,
                warped_left_bboxes,
                imageHeight,
                imageWidth,
                self.rpn_head.config["rpn_min_iou_with_target"]
            )
        else:
            # batch_hypotheses is given. Probably during ONNX export.
            rpn_cls_scores, rpn_bbox_preds = None, None

        for indexInBatch, left_bboxes_oneimage in enumerate(left_bboxes):
            # starttime = time.time()
            hypotheses = batch_hypotheses[indexInBatch]
            num_hypotheses = hypotheses["bboxes"].shape[0]
            if left_bboxes_oneimage.shape[0] == 0 or num_hypotheses == 0:
                # No detections in left
                list_sbboxes_priors.append(None)
                list_refined_right_bboxes.append(None)
                list_refined_right_scores.append(None)
                list_predicted_right_keypts.append(None)
                list_corresponding_leftdet_ids.append(None)
                continue

            left_priors = left_bboxes_oneimage[hypotheses["target_ids"]]
            right_priors = hypotheses["bboxes"]
            
            batch_number = indexInBatch * torch.ones((num_hypotheses)).unsqueeze(1).to(hypotheses["bboxes"].device)
            rois_right = right_priors.clone()
            rois_right = torch.cat((batch_number, rois_right), dim=1)

            right_roi_feats = self.bbox_roi_extractor(right_feats, rois_right)  # Note: Based on the bbox size to decide from which level to extract feats.
            # print("stereoNet time cost (until roi extract): {}".format(time.time() - starttime))
            # starttime = time.time()

            feats_hidden = right_roi_feats.flatten(1)
            for fc in self.shared_fcs:
                feats_hidden = self.relu(fc(feats_hidden))
            cls_score = self.fc_cls(feats_hidden)
            right_bboxes_refine = self.fc_reg(feats_hidden)
            right_keypts_pred = self.fc_keypts(feats_hidden)

            # starttime = time.time()
            # if self.logger is not None:
            #     for indexInstance in range(right_roi_feats.shape[0]):
            #         roi_feat_sample = torch.mean(right_roi_feats[indexInstance, :, :, :], dim=0).detach().cpu()
            #         roi_feat_sample = roi_feat_sample - roi_feat_sample.min()
            #         roi_feat_sample /= roi_feat_sample.max()
            #         self.logger.add_image("roi_feat_sample{}".format(indexInstance), roi_feat_sample)

            sbboxes_priors = torch.cat([left_priors, right_priors], dim=-1).view(1, num_hypotheses, 8)
            list_sbboxes_priors.append(sbboxes_priors)
            list_refined_right_bboxes.append(right_bboxes_refine)
            list_refined_right_scores.append(cls_score)
            list_predicted_right_keypts.append(right_keypts_pred)
            list_corresponding_leftdet_ids.append(hypotheses["target_ids"])
            # print("stereoNet time cost (one left dets proposals pass): {}".format(time.time() - starttime))

        return list_sbboxes_priors, list_corresponding_leftdet_ids, list_refined_right_bboxes, list_refined_right_scores, list_predicted_right_keypts, rpn_cls_scores, rpn_bbox_preds, batch_hypotheses

    def mask_lefttargets_withnogt(self, left_bboxes: List[Tensor], left_fg_mask: Tensor, left_nms_topk_mask: Tensor) -> List[Tensor]:
        left_bboxes_posgt = []
        for indexInBatch, bboxes_oneimage in enumerate(left_bboxes):
            pos_gt_mask_one = left_fg_mask[indexInBatch]
            pos_gt_mask_one = pos_gt_mask_one[left_nms_topk_mask[indexInBatch]]
            left_bboxes_posgt.append(bboxes_oneimage[pos_gt_mask_one])
        return left_bboxes_posgt

    def forward(
        self,
        right_event_voxel: Tensor,
        left_bboxes: List[Tensor],
        disp_prior: Tensor,
        batch_hypotheses_bboxes=None,
        batch_hypotheses_target_ids=None,
        labels=None,
        **kwargs
    ):
        """
        Note that gt bboxes in labels should align with left_bboxes and have same number (left detections are from detr).
        """
        if labels is not None:
            left_bboxes = self.mask_lefttargets_withnogt(left_bboxes, labels["left_fg_mask"], labels["left_nms_topk_mask"])

        if torch.onnx.is_in_onnx_export():
            assert batch_hypotheses_bboxes is not None and batch_hypotheses_target_ids is not None, "need given batch_hypotheses during onnx export."
            batch_hypotheses = [{"bboxes": batch_hypotheses_bboxes[indexInBatch], "target_ids": batch_hypotheses_target_ids[indexInBatch]} for indexInBatch in range(len(batch_hypotheses_bboxes))]
            preds = self.predict(right_event_voxel, left_bboxes, disp_prior, batch_hypotheses)
        else:
            preds = self.predict(right_event_voxel, left_bboxes, disp_prior)

        losses = None
        artifacts = None
        if labels is not None and not self.is_freeze:
            if kwargs["detector_format"] == "detr":
                losses = self.compute_loss_detrformat(preds, labels)
            elif kwargs["detector_format"] == "yolox":
                losses, artifacts = self.compute_loss_yoloxformat(preds, labels)
            elif kwargs["detector_format"] == "yolopose":
                losses, sbboxes, selected_keypts = self.compute_loss_yoloposeformat(preds, labels)
                # TODO: need a val_evaluator to count both precision and recall performance.
                right_gt_bboxes = [labels['stereo_objdet_targets']['bboxes'][labels['stereo_objdet_targets']['batch_idx'].view(-1) == indexBatch][:, [4, 1, 5, 3]] for indexBatch in range(len(sbboxes))]
                right_pred_bboxes = [sbboxes_oneimage[:, 4:] if sbboxes_oneimage is not None else None for sbboxes_oneimage in sbboxes]
                recallBatch = self.compute_recall(
                    right_pred_bboxes,
                    right_gt_bboxes,
                    iou_threshold=0.9,
                )  # recall @ 0.9 iou.
                artifacts = [sbboxes, recallBatch, selected_keypts]
        return preds, losses, artifacts
    
    def compute_recall(self, preds: list[Tensor], gt_bboxes: list[Tensor], iou_threshold: float) -> float:
        """
        Compute recall for the batch.
        Args:
            preds: list of shape [N, 4]. format [tl_x_r, tl_y_r, br_x_r, br_y_r] predicted right bboxes.
            gt_bboxes: list of shape [N, 4]. format [tl_x_r, tl_y_r, br_x_r, br_y_r] gt right bboxes.
            iou_threshold: iou threshold to determine a match.

        Returns:
            recall: float number.
        """
        assert len(preds) == len(gt_bboxes)
        total_gt = 0
        total_matched = 0
        for indexInBatch in range(len(preds)):
            if gt_bboxes[indexInBatch] is None:
                continue
            if preds[indexInBatch] is None:
                total_gt += gt_bboxes[indexInBatch].shape[0]
                continue
            ious_per_gt = compute_ious_pertarget(preds[indexInBatch], gt_bboxes[indexInBatch])  # [num_preds, num_gt]
            max_ious_per_gt = torch.max(ious_per_gt, dim=1)[0]
            matched = (max_ious_per_gt >= iou_threshold).sum().item()
            total_matched += matched
            total_gt += gt_bboxes[indexInBatch].shape[0]
        recall = total_matched / total_gt if total_gt > 0 else 0.0
        return recall

    def compute_loss_yoloposeformat(self, preds: Tuple[List, List, List], labels: Dict):
        """
        Design of the sampler:
        similar to SimOTA, using dynamic-k matching for bbox regression. For classification, use iou-encoded confidence as target for each candidate.

        Args:
            preds:
            labels: dict containing the following keys.
                left_fg_mask: mask for target assignment. (B, max_hypotheses)-> (4, 6615).
                left_target_gt_idx: for each foreground object, idx to the corresponding gt target. (B, max_hypotheses). need to be selected by fg_mask before use.
                left_nms_topk_mask: mask of filtering noisy left detections by nms and topk. (B, max_hypotheses).
                batch_img_metas: h, w of img.
                stereo_objdet_targets: dict
                    "bboxes": (N_allbatch, 6),
                    "cls":  (N_allbatch,),
                    "cls_leftdet": (N_allbatch,)
                    "keypoints": (N_allbatch, num_keypts*3), this is gt.
                    "keypoints_right": (N_allbatch, num_keypts*3), this is gt.
                    "batch_idx": (N_allbatch,). Index in this batch for this instance.
        """
        (
            list_sbboxes_priors,
            list_corresponding_leftdet_ids,
            list_refined_right_bboxes,
            list_right_scores_refine,
            list_predicted_right_keypts,
            rpn_cls_scores,
            rpn_bbox_preds,
            batch_hypotheses,
        ) = preds  # Note: number of positive in each can be different after nms.
        left_fg_mask, left_target_gt_idx, left_nms_topk_mask, stereo_objdet_targets, batch_img_metas = labels["left_fg_mask"], labels["left_target_gt_idx"], labels["left_nms_topk_mask"], labels["stereo_objdet_targets"], labels["batch_img_metas"]

        loss_dict = {}
        # rpn losses
        rpn_losses_cls, rpn_losses_bbox = self.rpn_head.compute_loss(rpn_cls_scores, rpn_bbox_preds, labels, batch_img_metas)
        loss_dict["loss_rrpn_cls"] = torch.stack(rpn_losses_cls).sum() * self._config["loss_cfg"]["rpn_losses_cls_weight"]
        loss_dict["loss_rrpn_bbox"] = torch.stack(rpn_losses_bbox).sum() * self._config["loss_cfg"]["rpn_losses_bbox_weight"]

        num_batch = len(list_sbboxes_priors)
        list_sbboxes_pred_refined = []
        list_right_keypts_pred = []
        iou_epsilon = 5e-2
        # right bboxes related loss
        for indexInBatch in range(num_batch):
            # starttime = time.time()
            mask_this_batch = stereo_objdet_targets["batch_idx"] == indexInBatch
            pos_gt_mask_one = left_fg_mask[indexInBatch]
            pos_gt_mask_one = pos_gt_mask_one[left_nms_topk_mask[indexInBatch]]
            num_pos_detections = pos_gt_mask_one.sum()
            
            if list_sbboxes_priors[indexInBatch] is None or pos_gt_mask_one.sum() == 0:
                # No detections from left
                list_sbboxes_pred_refined.append(None)
                list_right_keypts_pred.append(None)
                num_batch -= 1
                print("{}-th image in batch has no detections, skip.".format(indexInBatch))
                continue

            bbox_targetset_one = stereo_objdet_targets["bboxes"][mask_this_batch]
            idx_gt2left = left_target_gt_idx[indexInBatch][left_nms_topk_mask[indexInBatch]][pos_gt_mask_one]
            # TODO: need to discard hypotheses generated by leftdets that are not postive in pos_gt_mask_one.
            bbox_targets_one = bbox_targetset_one[idx_gt2left]  # one-to-one corresponding to leftdets with positive gt.
            idx_righthypos_corresponding_targets_ids = list_corresponding_leftdet_ids[indexInBatch].squeeze(0)
            if idx_righthypos_corresponding_targets_ids.dim() == 0:
                idx_righthypos_corresponding_targets_ids = idx_righthypos_corresponding_targets_ids.unsqueeze(0)

            rbboxes_targets = torch.concat([
                bbox_targets_one[:, 4].unsqueeze(-1),
                bbox_targets_one[:, 1].unsqueeze(-1),
                bbox_targets_one[:, 5].unsqueeze(-1),
                bbox_targets_one[:, 3].unsqueeze(-1)
            ], dim=1)  # (num_targets, 4)

            rbboxes_targets = rbboxes_targets[idx_righthypos_corresponding_targets_ids]
            rcls_targets = stereo_objdet_targets['cls'][mask_this_batch]
            # align gt with left detections' order
            rcls_targets = rcls_targets[idx_gt2left]  # (num_targets,)
            rcls_targets = rcls_targets[idx_righthypos_corresponding_targets_ids]  # (num_hypotheses,)
            sbboxes_priors = list_sbboxes_priors[indexInBatch].squeeze(0)  # shape [1, num_hypotheses, 8]. (tl_x, tl_y, br_x, br_y, tl_x_r, tl_y_r, br_x_r, br_y_r) format bbox, all in global scale.
            num_hypotheses_gtexist = sbboxes_priors.shape[0]
            right_priors = sbboxes_priors[:, 4:]  # (num_hypotheses_gtexist, 4)
            right_bboxes = list_refined_right_bboxes[indexInBatch].squeeze(0).view(num_hypotheses_gtexist, -1)
            right_scores = list_right_scores_refine[indexInBatch].squeeze(0).view(num_hypotheses_gtexist, -1)
            right_keypts = list_predicted_right_keypts[indexInBatch].squeeze(0).view(num_hypotheses_gtexist, -1)
            # print("stereoNet time cost (loss data uncollapse): {}".format(time.time() - starttime))
            # starttime = time.time()
            rbboxes_refined = right_bboxes.view(num_hypotheses_gtexist, self._config['num_classes'], -1)

            # advanced indexing to select bboxes for correct class.
            rbboxes_refined_selected = rbboxes_refined[
                torch.arange(num_hypotheses_gtexist),
                rcls_targets
            ]

            # decoding pred right bboxes
            rbboxes_refined_decoded = self.bbox_coder.decode(right_priors.view(-1, 4), rbboxes_refined_selected.view(-1, 4)).view(-1, 4)
            try:
                ious = self.batch_iou_calculator_simple(rbboxes_refined_decoded, rbboxes_targets)
            except:
                import IPython; import inspect; print('baodebug: file ({}) -- func ({})'.format(__file__, inspect.stack()[0].function)); IPython.embed()

            pos_mask, sampled_indices, rbboxes_refined_decoded_sampled, rbboxes_gt_sampled = self.batch_sampler(
                rbboxes_refined_decoded,
                ious,
                rbboxes_targets,
                self._config['r_iou_threshold'],
                self._config['candidates_k'],
            )
            rbboxes_refined_decoded_sampled = rbboxes_refined_decoded_sampled[:pos_mask.sum()]
            rbboxes_gt_sampled = rbboxes_gt_sampled[:pos_mask.sum()]

            # Note: encoding target and pred right bboxes before computing loss. See mmdet, bbox_head.py::BBoxHead::loss()
            # priors, gts -> encoded priors
            right_priors_sampled = right_priors[sampled_indices].clone()
            rbboxes_targets_sampled_encoded = self.bbox_coder.encode(right_priors_sampled, rbboxes_gt_sampled)
            rbboxes_targets_sampled_encoded[torch.isnan(rbboxes_targets_sampled_encoded)] = 0.0
            loss_rbbox_one = self.loss_rbbox(rbboxes_refined_selected[sampled_indices], rbboxes_targets_sampled_encoded)
            if "loss_rbbox" not in loss_dict:
                loss_dict["loss_rbbox"] = loss_rbbox_one
            else:
                loss_dict["loss_rbbox"] += loss_rbbox_one
            # print("stereoNet time cost (loss rbbox): {}".format(time.time() - starttime))
            # starttime = time.time()

            # right bboxes scores
            rbboxes_scores = right_scores.view(num_hypotheses_gtexist, -1)
            ious = ious.view(-1)
            rbboxes_cls_targets = rcls_targets.clone()
            foreground_mask = ious >= self._config.get('r_iou_bkg', 0.0)
            try:
                rbboxes_cls_targets[~foreground_mask] = self._config['num_classes']
            except:
                import IPython; import inspect; print('baodebug: file ({}) -- func ({})'.format(__file__, inspect.stack()[0].function)); IPython.embed()
            rbboxes_cls_targets_onehot = F.one_hot(rbboxes_cls_targets, num_classes=self._config['num_classes'] + 1).float()
            foreground_mask = foreground_mask.view(-1)
            rbboxes_cls_targets_onehot[foreground_mask, rbboxes_cls_targets[foreground_mask]] = ious[foreground_mask].clamp(iou_epsilon, 1.0)
            loss_rscore_one = self.loss_rscore(
                pred=rbboxes_scores,
                target=rbboxes_cls_targets_onehot
            )
            
            if "loss_rscore" not in loss_dict:
                loss_dict["loss_rscore"] = loss_rscore_one
            else:
                loss_dict["loss_rscore"] += loss_rscore_one
            # print("stereoNet time cost (loss rscore): {}".format(time.time() - starttime))
            # starttime = time.time()

            # loss for right keypts
            keypts_targetset_one = stereo_objdet_targets["keypoints_right"][mask_this_batch]
            # align gt order with left detections' order
            keypts_targets_one = keypts_targetset_one[idx_gt2left].view(num_pos_detections, -1)
            keypts_targets_one = keypts_targets_one[idx_righthypos_corresponding_targets_ids]

            # encode gt keypoints
            keypts_targets_one_encoded = keypts_targets_one.clone()
            for indexKeypt in range(self._config["max_num_keypoints"]):
                keypts_targets_one_encoded[:, (3 * indexKeypt):(2 + 3 * indexKeypt)] = EncodeKeypts(
                    right_priors[:, :],
                    keypts_targets_one_encoded[:, (3 * indexKeypt):(2 + 3 * indexKeypt)],
                )
            # select keypooints by predicted class labels
            right_keypts_pred = right_keypts.view(num_hypotheses_gtexist, self._config['num_classes'], -1)
            right_keypts_selected = right_keypts_pred[
                torch.arange(num_hypotheses_gtexist),
                rcls_targets
            ]

            right_keypts_selected_pos = right_keypts_selected[sampled_indices].view(-1, self._config["max_num_keypoints"], 3)  # (num_pos, num_keypts, 3)
            keypts_targets_selected_encoded = keypts_targets_one_encoded[sampled_indices].view(-1, self._config["max_num_keypoints"], 3)  # (num_pos, num_keypts, 3)
            right_priors_selected_xywh = xyxy2xywh(right_priors[sampled_indices])
            areas = right_priors_selected_xywh[:, 2:].prod(1, keepdim=True)
            kpt_mask = keypts_targets_selected_encoded.view(-1, self._config["max_num_keypoints"], 3)[..., 2] > 0
            loss_rkeypts = self.loss_keypoint(
                right_keypts_selected_pos,
                keypts_targets_selected_encoded,
                kpt_mask,
                areas
            )
            loss_rkeypts_obj = self.loss_bce_pose(
                right_keypts_selected_pos[..., 2],
                kpt_mask.float()
            )

            if "loss_rkeypts" not in loss_dict:
                loss_dict["loss_rkeypts"] = loss_rkeypts
                loss_dict["loss_rkeypts_obj"] = loss_rkeypts_obj
            else:
                loss_dict["loss_rkeypts"] += loss_rkeypts
                loss_dict["loss_rkeypts_obj"] += loss_rkeypts_obj
            # print("stereoNet time cost (loss rkeypts & bce_pose): {}".format(time.time() - starttime))
            # starttime = time.time()

            # select best right bboxes and keypts for visualization.
            (
                mask_nonbackground,
                refined_sbboxes_nobkg,
                refined_right_scored_pred,
                right_keypts_pred_nobkg
            ) = self.extract_inference_results(
                sbboxes_priors,
                rbboxes_refined,
                rbboxes_scores,
                right_keypts_pred
            )
            if refined_sbboxes_nobkg is not None:
                # select only the highest scoring right bboxes and keypoints to pair with left
                selected_refined_sbboxes_nobkg = []
                selected_right_keypts_pred_nobkg = []
                try:
                    indices_leftDet = torch.unique(idx_righthypos_corresponding_targets_ids[mask_nonbackground])
                except:
                    import IPython; import inspect; print('baodebug: file ({}) -- func ({})'.format(__file__, inspect.stack()[0].function)); IPython.embed()
                for indexLeftDet in indices_leftDet:
                    mask_indexleftdet = idx_righthypos_corresponding_targets_ids[mask_nonbackground] == indexLeftDet
                    index_highest_score = torch.argmax(refined_right_scored_pred[mask_indexleftdet])
                    selected_refined_sbboxes_nobkg.append(refined_sbboxes_nobkg[mask_indexleftdet][index_highest_score].unsqueeze(0))
                    selected_right_keypts_pred_nobkg.append(right_keypts_pred_nobkg[mask_indexleftdet][index_highest_score].unsqueeze(0))

                # # ----------- debug code -----------
                # refined_sbboxes_nobkg = sbboxes_priors
                # right_keypts_pred_nobkg = right_keypts_pred[:, 0, :].view(-1, self._config['max_num_keypoints'], 3)
                # mask_nonbackground = torch.ones_like(mask_nonbackground, dtype=torch.bool)
                # # ----------- debug code -----------
                list_sbboxes_pred_refined.append(torch.cat(selected_refined_sbboxes_nobkg, dim=0))
                list_right_keypts_pred.append(torch.cat(selected_right_keypts_pred_nobkg, dim=0))
            else:
                list_sbboxes_pred_refined.append(None)
                list_right_keypts_pred.append(None)

        if "loss_rbbox" in loss_dict:
            loss_dict["loss_rbbox"] /= num_batch
            loss_dict["loss_rbbox"] *= self._config["loss_cfg"]["rbbox_loss_weight"]

        if "loss_rscore" in loss_dict:
            loss_dict["loss_rscore"] /= num_batch
            loss_dict["loss_rscore"] *= self._config["loss_cfg"]["rscore_loss_weight"]

        if "loss_rkeypts" in loss_dict:
            loss_dict["loss_rkeypts"] /= num_batch
            loss_dict["loss_rkeypts"] *= self._config["loss_cfg"]["rkeypts_loss_weight"]

        if "loss_rkeypts_obj" in loss_dict:
            loss_dict["loss_rkeypts_obj"] /= num_batch
            loss_dict["loss_rkeypts_obj"] *= self._config["loss_cfg"]["rkeypts_obj_loss_weight"]

        return loss_dict, list_sbboxes_pred_refined, list_right_keypts_pred

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

        try:
            rselect_mask, rbboxes_refined_selected, rbboxes_targets_selected = self.batch_assigner(
                rbboxes_refined,
                ious,
                rbboxes_targets,
                self._config['r_iou_threshold'],
                self._config['candidates_k']
            )
        except:
            import IPython; import inspect; print('baodebug: file ({}) -- func ({})'.format(__file__, inspect.stack()[0].function)); IPython.embed()
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
    def extract_inference_results(
        self,
        sbboxes_priors: Tensor,
        refined_right_bboxes: Tensor,
        refined_right_scores: Tensor,
        right_keypts_pred: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Extract inference results from the model's predictions.
        Args:
            sbboxes_priors: (num_hypotheses_gtexist, 8) shape, containing the right priors from preivous network.
            refined_right_bboxes: (num_hypotheses_gtexist, num_classes, 4) shape.
            refined_right_scores: (num_hypotheses_gtexist, num_classes+1) shape.
            right_keypts_pred: (num_hypotheses_gtexist, num_classes, num_keypts*3) shape.
        
        Returns:
            mask_nonbackground: (num_detections,) shape.
            refined_sbboxes: (<num_detections>, 8) shape, <num_detections> == mask_nonbackground.sum().
            refined_right_scores_pred: (<num_detections>, 1) shape.
            right_keypts_pred: (<num_detections>, num_keypts*3) shape.
        """
        num_hypotheses_gtexist = sbboxes_priors.shape[:2]
        num_classes = refined_right_bboxes.shape[-2]

        best_scores_of_classes, best_class_labels = torch.max(F.softmax(refined_right_scores, dim=-1), dim=-1)
        class_labels_pred = best_class_labels
        mask_nonbackground = class_labels_pred != num_classes
        if mask_nonbackground.sum() == 0:
            print("Warning: all detections are classified as bkground, no valid detections.")
            return mask_nonbackground, None, None, None
        else:
            sbboxes_priors_nobkg = sbboxes_priors[mask_nonbackground]
            refined_sbboxes_nobkg = sbboxes_priors_nobkg.view(-1, 8).clone()
            refined_right_bboxes_nobkg = refined_right_bboxes[mask_nonbackground]
            right_keypts_pred_nobkg = right_keypts_pred[mask_nonbackground]
            class_labels_pred_nobkg = class_labels_pred[mask_nonbackground]
            # select the highest score as confidence score
            refined_right_scores_pred = best_scores_of_classes[mask_nonbackground]
            # select data based on the class label
            refined_right_bboxes_nobkg = refined_right_bboxes_nobkg[
                torch.arange(mask_nonbackground.sum()),
                class_labels_pred_nobkg
            ]
            right_priors_nobkg = sbboxes_priors_nobkg[..., 4:]  # (num_hypotheses_nobkg, 4)
            refined_right_bboxes_nobkg_decoded = self.bbox_coder.decode(right_priors_nobkg, refined_right_bboxes_nobkg.view(-1, 4))
            refined_sbboxes_nobkg[:, 4:] = refined_right_bboxes_nobkg_decoded
            # select best keypoints
            right_keypts_pred_nobkg = right_keypts_pred_nobkg[
                torch.arange(mask_nonbackground.sum()),
                class_labels_pred_nobkg
            ]
            for indexKeypt in range(self._config["max_num_keypoints"]):
                right_keypts_pred_nobkg[:, (3 * indexKeypt):(2 + 3 * indexKeypt)] = DecodeKeypts(
                    right_priors_nobkg,
                    right_keypts_pred_nobkg[:, (3 * indexKeypt):(2 + 3 * indexKeypt)],
                )
            return mask_nonbackground, refined_sbboxes_nobkg, refined_right_scores_pred, right_keypts_pred_nobkg.view(mask_nonbackground.sum(), -1, 3)

    @torch.no_grad()
    def batch_keypts_distance_calculator_simple(self, keypts_preds: Tensor, keypts_ref: Tensor):
        """
        compute euclidian distances between each pair of keypoints and the only ref pair of keypoints.
        Args:
            keypts_preds: [B. num_grids, 3*num_keypts]. keypts format(keypt0_x, keypt0_y, visibility, keypt1_x, keypt1_y, visibility, ...)
            keypts_ref: [B, 1, 3*num_keypts].

        Returns:
            distances: [B, num_grids]
            indices_best_right: [B]
        """
        assert keypts_ref.shape[1] == 1, "keypts_ref should have only one at each instance."
        num_grids = keypts_preds.shape[1]
        keypts_ref = keypts_ref.expand(-1, num_grids, -1)
        num_keypts = self._config['max_num_keypoints']
        distances = None
        for iKeypt in range(num_keypts):
            if distances is None:
                distances = torch.norm(keypts_ref[..., [0, 1]] - keypts_preds[..., [0, 1]], dim=-1)
            else:
                distances += torch.norm(keypts_ref[..., [3 * iKeypt, 1 + 3 * iKeypt]] - keypts_preds[..., [3 * iKeypt, 1 + 3 * iKeypt]], dim=-1)
        return distances, torch.min(distances, dim=-1)[1]

    @torch.no_grad()
    def batch_iou_calculator_simple(self, bboxes_preds: Tensor, bboxes_ref: Tensor):
        """
        compute IoU between each preds group and the corresponding ref bboxes.
        Args:
            bboxes_preds: [N, 4]. box format (tl_x, tl_y, br_x, br_y). N is the number of detection instances.
            bboxes_ref: [N, 4].

        Returns: 
            ious: [N,]
        """
        intersectionBox_tl_x = torch.max(torch.stack((bboxes_preds[..., 0], bboxes_ref[..., 0]), dim=-1), dim=-1)[0]
        intersectionBox_br_x = torch.min(torch.stack((bboxes_preds[..., 2], bboxes_ref[..., 2]), dim=-1), dim=-1)[0]
        
        ws = torch.clamp(intersectionBox_br_x - intersectionBox_tl_x, min=0, max=None)
        intersectionBox_tl_y = torch.max(torch.stack((bboxes_preds[..., 1], bboxes_ref[..., 1]), dim=-1), dim=-1)[0]
        intersectionBox_br_y = torch.min(torch.stack((bboxes_preds[..., 3], bboxes_ref[..., 3]), dim=-1), dim=-1)[0]
        hs = torch.clamp(intersectionBox_br_y - intersectionBox_tl_y, min=0, max=None)
        intersectionAreas = ws * hs
        unionAreas = (bboxes_ref[..., 2] - bboxes_ref[..., 0]) * (bboxes_ref[..., 3] - bboxes_ref[..., 1]) + (bboxes_preds[..., 2] - bboxes_preds[..., 0]) * (bboxes_preds[..., 3] - bboxes_preds[..., 1])
        unionAreas = unionAreas - intersectionAreas
        ious = intersectionAreas / unionAreas
        return ious

    def batch_sampler(self, bboxes_preds: Tensor, iou_scores: Tensor, batch_gt_bboxes: Tensor, iou_thres: float, candidates_k: int):
        """
        for each gt, there are N candidates. Find the best candidates based on IoU_thres and candidates_k.
        If candidates within IoU_thres are less than candidates_k, 0 pad them.

        Args:
            bboxes_preds: shape (N, 4)
            iou_scores: shape (N,)
            batch_gt_bboxes: shape (N, 4)

        Returns:
            pos_mask: shape (N, num_grids), mask for positive samples.
            bboxes_preds_selected: shape (N, k, 4). Note some of the k elements can be just 0s.
            bboxes_gt_selected: shape (N, k, 4). Note some of the k elements can be just 0s.
        """
        num_detections = bboxes_preds.shape[0]
        with torch.no_grad():
            candidates_k = min(candidates_k, num_detections)
            topk_scores, topk_indices = torch.topk(iou_scores, candidates_k, dim=0)
            valid_mask = topk_scores > iou_thres
            valid_mask[:1] = True  # Note: make sure at least 1 candidate for each gt; torch.topk sorted topk_scores and the first one is sorted to be the best.
            topk_indices = topk_indices.masked_fill(~valid_mask, num_detections)

        pseudo_bboxes = torch.zeros(1, 4, dtype=bboxes_preds.dtype, device=bboxes_preds.device)
        bboxes_preds_padded = torch.cat([bboxes_preds, pseudo_bboxes], dim=0)
        bboxes_preds_selected = torch.gather(bboxes_preds_padded, 0, topk_indices.unsqueeze(-1).expand(-1, 4))
        bboxes_gt_selected = torch.cat([batch_gt_bboxes, pseudo_bboxes], dim=0)
        bboxes_gt_selected = bboxes_gt_selected[topk_indices].masked_fill(~valid_mask.unsqueeze(-1).expand(-1, 4), 0)

        # make sure at least one candidate for each gt
        # candidates_mask[torch.arange(0, num_detections, device=candidates_mask.device), topk_indices[:, 0]] = True
        pos_mask = torch.zeros_like(iou_scores, dtype=torch.bool)  # build a new mask with only topk being True.
        valid_mask = topk_indices < num_detections
        pos_mask[topk_indices[valid_mask]] = True

        return pos_mask, topk_indices[valid_mask], bboxes_preds_selected, bboxes_gt_selected
    
    def batch_assigner_keypts(self, keypts_preds: Tensor, distances: Tensor, batch_gt_keypts: Tensor, distance_threshold: float, candidates_k: int):
        """
        for each gt, there are N candidates. Find the best candidates based on distance_threshold and candidates_k.
        If candidates within distance_threshold are less than candidates_k, 0 pad them.

        Args:
            keypts_preds: shape (B, numGrids, 3*num_keypts)
            distances: shape (B, numGrids)
            batch_gt_keypts: shape (B, 3*num_keypts)
        Returns:
            candidates_mask: shape (B, numGrids). boolean mask.
            keypts_preds_selected: shape (B, k, 3*num_keypts). Note some of the k elemenets can be just 0s.
            bboxes_targets_selected: shape (B, k, 3*num_keypts). Note some of the k elements can be just 0s.
        """
        dim_keypts_preds = self._config['max_num_keypoints'] * 3

        keypts_targets_selected = batch_gt_keypts.view(-1, 1, dim_keypts_preds).repeat(1, candidates_k, 1)
        batch_size, num_grids = keypts_preds.shape[:2]

        with torch.no_grad():
            topk_distances, topk_indices = torch.topk(distances, candidates_k, dim=1, largest=False)
            valid_mask = topk_distances < distance_threshold
            valid_mask[:, 0] = True  # Note: make sure at least one candidate for each gt.
            topk_indices = topk_indices.masked_fill(~valid_mask, num_grids)

        pseudo_keypts = torch.zeros(batch_size, 1, dim_keypts_preds, dtype=keypts_preds.dtype, device=keypts_preds.device)
        keypts_preds_padded = torch.cat([keypts_preds, pseudo_keypts], dim=1)
        keypts_preds_selected = torch.gather(keypts_preds_padded, 1, topk_indices.unsqueeze(-1).expand(-1, -1, dim_keypts_preds))
        keypts_targets_selected = keypts_targets_selected.masked_fill(~valid_mask.unsqueeze(-1).expand(-1, -1, dim_keypts_preds), 0)

        # make sure at least one candidate for each gt
        # candidates_mask[torch.arange(0, batch_size, device=candidates_mask.device), topk_indices[:, 0]] = True
        candidates_mask = torch.zeros_like(distances, dtype=torch.bool)  # build a new mask with only topk being True.
        valid_mask = topk_indices < num_grids
        for indexInBatch in range(batch_size):
            candidates_mask[indexInBatch, topk_indices[indexInBatch][valid_mask[indexInBatch]]] = True
        
        return candidates_mask, keypts_preds_selected, keypts_targets_selected


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
