import torch.nn as nn
import torch
from torch import Tensor
from typing import Sequence

from ...utils.misc import multi_apply


class Cylinder5DDetectionHead(nn.Module):
    # network layers
    _multi_level_cls_convs = None
    _multi_level_reg_convs = None
    _multi_level_keypts1_convs = None  # Note: two keypoints needed. One at object center, one at object top center.
    _multi_level_keypts2_convs = None

    def __init__(self, num_scales):
        pass

    def _init_layers(self) -> None:
        """
        initialize the head for all levels of feature maps
        """

    def forward_objdet_single(
            self,
            left_feat: Tensor,
            right_feat: Tensor,
            cls_conv: nn.Module,
            reg_conv: nn.Module,
        ):
        """
        forward feature of a single scale level
        """
        # cls, stereo bbox, objectness prediction, similar to yolo
        cls_feat = cls_con

        
    def forward_keypts_single(
            self,
            left_feat: Tensor,
            keypts_conv: nn.Module
    ):
        """
        predict keypoint bias to the grid center, e.g. [delta_x, delta_y]
        """
        # keypoint pred. manually crop gt patches from left_img and use them to predict keypoints.
        # 
        pass

    def calculate_loss_by_feat(
        self,
        cls_scores: Sequence[Tensor],
        stereo_bboxes_preds: Sequence[Tensor],
        objectness_preds: Sequence[Tensor],
        disparity_pyramid: Sequence[Tensor],
        batch_gt_labels: Sequence[Tensor],
        batch_img_metas: Sequence[dict]
    ):
        pass