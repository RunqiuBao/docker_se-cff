import torch.nn as nn
import torch
from torch import Tensor
import torch.nn.functional as F
from torchvision.ops import roi_align
from typing import Sequence, Tuple

from ...utils.misc import multi_apply


class Cylinder5DDetectionHead(nn.Module):
    # network layers
    _multi_level_cls_convs = None
    _multi_level_reg_convs = None
    _multi_level_conv_cls = None
    _multi_level_conv_reg = None
    _multi_level_conv_obj = None
    _multi_level_keypts1_convs = None  # Note: two keypoints needed. One at object center, one at object top center.
    _multi_level_keypts2_convs = None

    _config = None

    def __init__(
        self
    ):
        self._config = {
            'num_classes': 1,
            'in_channels': 128,
            'feat_channels': 256,
            'num_stacked_convs': 2,
            'strides': [8, 16, 32],
            'norm_cfg': dict(type='BN', momentum=0.03, eps=0.001),
            'act_cfg': dict(type='ReLU'),
            # keypts pred
            'aligned_roi_size': (14, 14)
        }
        self._init_layers()
        self.__init_weights()

    def _init_layers(self) -> None:
        """
        initialize the head for all levels of feature maps
        """
        self._multi_level_cls_convs = nn.ModuleList()
        self._multi_level_reg_convs = nn.ModuleList()
        self._multi_level_conv_cls = nn.ModuleList()
        self._multi_level_conv_reg = nn.ModuleList()
        self._multi_level_conv_obj = nn.ModuleList()
        self._multi_level_keypts1_convs = nn.ModuleList()
        self._multi_level_keypts2_convs = nn.ModuleList()
        for _ in self._config['strides']:
            self._multi_level_cls_convs.append(
                self._build_objdet_convs(
                    in_channels=self._config['in_channels'] * 2,  # Note: concat left and right feature maps
                    feat_channels=self._config['feat_channels'],
                    norm_eps=self._config['norm_cfg']['eps'],
                    norm_momentum=self._config['norm_cfg']['momentum'],
                    act_type=self._config['act_cfg']['type']))
            self._multi_level_reg_convs.append(
                self._build_objdet_convs(
                    in_channels=self._config['in_channels'] * 2,
                    feat_channels=self._config['feat_channels'],
                    norm_eps=self._config['norm_cfg']['eps'],
                    norm_momentum=self._config['norm_cfg']['momentum'],
                    act_type=self._config['act_cfg']['type']))
            conv_cls = nn.Conv2d(self._config['feat_channels'], self._config['num_classes'], 1)
            conv_reg = nn.Conv2d(self._config['feat_channels'], 6, 1)
            conv_obj = nn.Conv2d(self._config['feat_channels'], 1, 1)
            self._multi_level_conv_cls.append(conv_cls)
            self._multi_level_conv_reg.append(conv_reg)
            self._multi_level_conv_obj.append(conv_obj)
            self._multi_level_keypts1_convs.append(self._build_keypts_convs(
                in_channels=self._config['feat_channels'],
                feat_channels=self._config['feat_channels'],
                num_classes=self._config['num_classes']
            ))
            self._multi_level_keypts2_convs.append(self._build_keypts_convs(
                in_channels=self._config['feat_channels'],
                feat_channels=self._config['feat_channels'],
                num_classes=self._config['num_classes']
            ))

    def __init_weights(self):
        """
        all conv2d need weights initialization
        """
        for module_list in [self._multi_level_cls_convs, self._multi_level_reg_convs, self._multi_level_conv_cls, self._multi_level_conv_reg, self._multi_level_conv_obj, self._multi_level_keypts1_convs, self._multi_level_keypts2_convs]:
            for module in module_list:
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5), mode='fan_in', nonlinearity='leaky_relu')
        prior_prob = 0.01
        bias_init = float(-numpy.log((1 - prior_prob) / prior_prob))
        for module_list in [self._multi_level_conv_cls, self._multi_level_conv_obj]:
            for module in module_list:
                module.bias.data.fill_(bias_init)

    @staticmethod
    def _build_keypts_convs(
        in_channels: int,
        feat_channels: int,
        num_classes: int
    ):
        """
        refer to mask-rcnn mask-prediction head.
        """
        stacked_convs = []
        in_channels_temp = in_channels
        for indexConvLayer in range(4):
            stacked_convs.append(
                nn.Conv2d(
                    in_channels_temp,
                    feat_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            in_channels_temp = feat_channels
            stacked_convs.append(nn.ReLU(inplace=True))
        stacked_convs.append(
            nn.ConvTranspose2d(
                feat_channels,
                feat_channels,
                kernel_size=4,
                stride=2,
                padding=1
            ))
        stacked_convs.append(nn.ReLU(inplace=True))
        stacked_convs.append(
            nn.Conv2d(
                feat_channels,
                num_classes,
                kernel_size=1,
                stride=1,
                padding=1
            ))

        return nn.Sequential(*stacked_convs)

    @staticmethod
    def _build_objdet_convs(
        in_channels: int,
        feat_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        norm_eps: float = 1e-5,
        norm_momentum: float = 0.1,
        act_type: str = 'ReLU'
    ) -> nn.Sequential:
        """
        initialize conv (conv, bn, act) layes for a single level head.
        """
        stacked_convs = []
        for indexConvLayer in range(self._config['num_stacked_convs']):
            stacked_convs.append(
                nn.Conv2d(
                    in_channels,
                    feat_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding
                )
            )
            stacked_convs.append(nn.BatchNorm2d(feat_channels, eps=norm_eps, momentum=norm_momentum))
            stacked_convs.append(getattr(nn, act_type)())
        return nn.Sequential(*stacked_convs)

    def forward_objdet_single(
            self,
            left_feat: Tensor,
            right_feat: Tensor,
            cls_convs: nn.Module,
            reg_convs: nn.Module,
            conv_cls: nn.Module,
            conv_reg: nn.Module,
            conv_obj: nn.Module
        ) -> Tuple(Tensor, Tensor, Tensor):
        """
        forward feature of a single scale level
        Args:
            left_feat: (b, c, h, w) tensor from left camera of stereo pair.
            ...
        """
        x = torch.cat((left_feat, right_feat), 1)
        # cls, stereo bbox, objectness prediction, similar to yolo
        cls_feat = cls_convs(x)
        reg_feat = reg_convs(x)

        cls_score = conv_cls(cls_feat)  # (b, num_classes, h, w) shape. probabilities of class of the bbox at each grid.
        bbox_pred = conv_reg(reg_feat)  # (b, 6, h, w) shape. each grid contains one predicted bbox.
        objectness = conv_obj(reg_feat)  # (b, 1, h, w) shape, confidence of whether the object in the bbox is an object.

        return cls_score, bbox_pred, objectness

    def forward_keypts_single(
            self,
            left_feat: Tensor,
            bbox_pred: Tensor,
            keypts_convs: nn.Module,
    ):
        """
        bbox_pred is top 100 bbox predictions from forward_objdet_single. Same as mask-rcnn.
        predict keypoint bias to the grid center, e.g. [delta_x, delta_y].
        predict one keypoint for each class, shape of output is (b, 100, num_classes, 2). only take the k-th keypoint, k is the index of class of the bbox.
        Args:
            left_feat: (b, N, h, w) tensor. N is the number of channels of the feature map. 
            bbox_pred: (b, 100, 4) tensor. predicted bbox of objects. 100 is the number of bboxes. 4 channels are [tl_x, tl_y, br_x, br_y] format bbox.
            cls_score: (b, 100, num_classes) tensor.
            ... 

        Return:
            keypts_pred: (b, 100, num_classes, 2) tensor. predicted keypoint bias (normalized) to the bbox top left corner.
        """
        bbox_pred_list = []
        for bbox_pred_per_img in bbox_pred:
            bbox_pred_list.append(bbox_pred_per_img)
        aligned_rois = roi_align(left_feat, bbox_pred_list, output_size=self._config['aligned_roi_size'])  # Note: output shape is (b*100, N, 14, 14)
        keypts_feat = keypts_convs(aligned_rois)  # Note: (b*100, num_of_class, 28, 28)
        keypts_prob_x = torch.argmax(F.softmax(keypts_feat.sum(dim=-2), dim=-1), dim=-1).unsqueeze(-1) / keypts_feat.shape[-1]  # Note: shape is (b*100, num_classes, 1)
        keypts_prob_y =torch.argmax(F.softmax(keypts_feat.sum(dim=-1), dim=-1), dim=-1).unsqueeze(-1) / keypts_feat.shape[-2]
        keypts_pred = torch.cat((keypts_prob_x, keypts_prob_y), dim=-1)  # Note: shape is (b*100, num_classes, 2)
        batch_size, num_bboxes = bbox_pred.shape[:2]
        num_classes, num_pts_dimension = keypts_pred.shape[-2:]
        return keypts_pred.view(batch_size, num_bboxes, num_classes, num_pts_dimension)

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

    def predict_by_feat(
        self
    ):
        pass