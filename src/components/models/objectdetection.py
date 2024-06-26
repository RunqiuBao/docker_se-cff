import torch.nn as nn
import torch
from torch import Tensor
import torch.nn.functional as F
from torchvision.ops import roi_align
from typing import Sequence, Tuple, List, Dict, Union
import math
import numpy
import time
import copy

from mmdet.models.task_modules.prior_generators import MlvlPointGenerator
from mmdet.models.task_modules.samplers import PseudoSampler
from mmdet.models.roi_heads.roi_extractors.single_level_roi_extractor import SingleRoIExtractor
from mmengine.structures import InstanceData
from mmdet.structures.bbox import cat_boxes
from mmdet.registry import TASK_UTILS, MODELS
from mmdet.utils import reduce_mean
from mmcv.ops import batched_nms

from .utils.misc import multi_apply, SelectByIndices, ComputeSoftArgMax1d, freeze_module_grads, DetachCopyNested
from . import losses


class Cylinder5DDetectionHead(nn.Module):

    def __init__(self, net_cfg: Dict, loss_cfg: Dict, is_distributed: bool, logger=None):
        super().__init__()
        self._config = net_cfg
        self.logger = logger
        self.strides = [8, 16, 32]

        self._init_layers()
        self.__init_weights()
        # objdet tools
        self.assigner = TASK_UTILS.build({'type': 'SimOTAAssigner', 'center_radius': 2.5})
        self.sampler = PseudoSampler()
        self.bbox_roi_extractor = MODELS.build(
            {
                'type': 'SingleRoIExtractor',
                'roi_layer': {'type': 'RoIAlign', 'output_size': self._config['right_roi_feat_size'], 'sampling_ratio': 0},
                'out_channels': 128,
                'featmap_strides': self.strides
            }
        )
        # loss preparation
        self.loss_obj = MODELS.build({'type': 'CrossEntropyLoss',
                                      'use_sigmoid': True,
                                      'reduction': 'sum',
                                      'loss_weight': 1.0})
        self.loss_cls = MODELS.build({'type': 'CrossEntropyLoss',
                                      'use_sigmoid': True,
                                      'reduction': 'sum',
                                      'loss_weight': 1.0})
        self.loss_bbox = MODELS.build({'type': 'IoULoss',
                                        'mode': 'square',
                                        'eps': 1e-16,
                                        'reduction': 'sum',
                                        'loss_weight': 5.0})
        self.loss_rbbox = MODELS.build({'type': 'IoULoss',  # Right side bbox
                                        'mode': 'square',
                                        'eps': 1e-16,
                                        'reduction': 'sum',
                                        'loss_weight': 1.0})
        # Need to consider keypts in loss_bbox as well.
        self.loss_keypt1 = MODELS.build({'type': 'CrossEntropyLoss',
                                    'use_sigmoid': True,
                                    'reduction': 'sum',
                                    'loss_weight': 1.0})
        self.loss_keypt2 = MODELS.build({'type': 'CrossEntropyLoss',
                                    'use_sigmoid': True,
                                    'reduction': 'sum',
                                    'loss_weight': 1.0})
        # points generator for multi-level (mlvl) feature maps
        self.prior_generator = MlvlPointGenerator(self.strides, offset=0)

    def _init_layers(self) -> None:
        """
        initialize the head for all levels of feature maps
        """ 
        # stereo det subnet
        self.right_bbox_refiner = self._build_bbox_refiner_convs(
            self._config['in_channels'],
            self._config['feat_channels'],
            output_logits=2,
            final_featmap_size=self._config['right_roi_feat_size'] // 2
        )

        # Note: two keypoints needed. One at object center, one at object top center.
        self.keypt1_predictor = self._build_keypts_convs(
            in_channels=self._config["in_channels"],
            feat_channels=self._config["feat_channels"],
            num_classes=self._config["num_classes"],
        )
        self.keypt2_predictor = self._build_keypts_convs(
            in_channels=self._config["in_channels"],
            feat_channels=self._config["feat_channels"],
            num_classes=self._config["num_classes"],
        )
        # ========== multi level convs init ==========
        self._multi_level_cls_convs = nn.ModuleList()
        self._multi_level_reg_convs = nn.ModuleList()
        self._multi_level_conv_cls = nn.ModuleList()
        self._multi_level_conv_reg = nn.ModuleList()
        self._multi_level_conv_obj = nn.ModuleList()
        for indexLevel in range(len(self.strides)):
            self._multi_level_cls_convs.append(
                self._build_objdet_convs(
                    num_stacked_convs=self._config['num_stacked_convs'],
                    in_channels=self._config['in_channels'],  # Note: cost volume channel size
                    feat_channels=self._config["feat_channels"],
                    norm_eps=self._config["norm_cfg"]["eps"],
                    norm_momentum=self._config["norm_cfg"]["momentum"],
                    act_type=self._config["act_cfg"]["type"],
                )
            )
            self._multi_level_reg_convs.append(
                self._build_objdet_convs(
                    num_stacked_convs=self._config['num_stacked_convs'],
                    in_channels=self._config['in_channels'],
                    feat_channels=self._config["feat_channels"],
                    norm_eps=self._config["norm_cfg"]["eps"],
                    norm_momentum=self._config["norm_cfg"]["momentum"],
                    act_type=self._config["act_cfg"]["type"],
                )
            )
            conv_cls = nn.Conv2d(
                self._config["feat_channels"], self._config["num_classes"], 1
            )
            conv_reg = nn.Conv2d(self._config["feat_channels"], 4, 1)
            conv_obj = nn.Conv2d(self._config["feat_channels"], 1, 1)
            self._multi_level_conv_cls.append(conv_cls)
            self._multi_level_conv_reg.append(conv_reg)
            self._multi_level_conv_obj.append(conv_obj)

    def __init_weights(self):
        """
        all conv2d need weights initialization
        """
        for module_list in [
            self._multi_level_cls_convs,
            self._multi_level_reg_convs,
            self._multi_level_conv_cls,
            self._multi_level_conv_reg,
            self._multi_level_conv_obj
        ]:
            for module in module_list:
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_uniform_(
                        module.weight,
                        a=math.sqrt(5),
                        mode="fan_in",
                        nonlinearity="leaky_relu"
                    )

        for subnet in [self.right_bbox_refiner, self.keypt1_predictor, self.keypt2_predictor]:
            for module in subnet.modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    nn.init.kaiming_uniform_(
                        module.weight,
                        a=math.sqrt(5),
                        mode="fan_in",
                        nonlinearity="leaky_relu"
                    )

        prior_prob = 0.01
        bias_init = float(-numpy.log((1 - prior_prob) / prior_prob))
        for module_list in [self._multi_level_conv_cls, self._multi_level_conv_obj]:
            for module in module_list:
                module.bias.data.fill_(bias_init)

    @staticmethod
    def _build_keypts_convs(in_channels: int, feat_channels: int, num_classes: int):
        """
        refer to mask-rcnn mask-prediction head.
        """
        stacked_convs = []
        in_channels_temp = in_channels
        for indexConvLayer in range(4):
            stacked_convs.append(
                nn.Conv2d(
                    in_channels_temp,feat_channels, kernel_size=3, stride=1, padding=1
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
    def _build_objdet_convs(
        num_stacked_convs: int,
        in_channels: int,
        feat_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        norm_eps: float = 1e-5,
        norm_momentum: float = 0.1,
        act_type: str = "ReLU",
    ) -> nn.Sequential:
        """
        initialize conv (conv, bn, act) layes for a single level head.
        """
        stacked_convs = []
        for indexConvLayer in range(num_stacked_convs):
            stacked_convs.append(
                nn.Conv2d(
                    in_channels if indexConvLayer == 0 else feat_channels,
                    feat_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )
            stacked_convs.append(
                nn.BatchNorm2d(feat_channels, eps=norm_eps, momentum=norm_momentum)
            )
            stacked_convs.append(getattr(nn, act_type)())
        return nn.Sequential(*stacked_convs)
    
    @staticmethod
    def _build_bbox_refiner_convs(
        in_channels: int,
        feat_channels: int,
        output_logits: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        final_featmap_size: int = 7
    ) -> nn.Sequential:
        """
        For left, output logits are [delta_x, delta_y, w, h], w and h are relative values to bbox size;
        For right, output two logits are [delta_x, w].
        """
        bbox_refiner = nn.Sequential(
            nn.Conv2d(in_channels, feat_channels, kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(feat_channels * final_featmap_size * final_featmap_size, 4096),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(4096, output_logits)
        )
        return bbox_refiner

    def forward(
        self,
        left_feature: List[Tensor],
        right_feature: List[Tensor],
        disparity_prior: Tensor,
        batch_img_metas: Dict,
        gt_labels: Tensor = None,
    ):
        """
        Args:
            left_feature: multi-level features from left image input.
            right_feature: multi-level features from right image input.
            gt_labels: dict. it contains 'bboxes' and 'labels' keys.
        Returns:
            Tuple[List]: A tuple of multi-level classification scores, bbox predictions, keypts predcitions, objectnesses.
        """
        starttime = time.time()
        preds_items_multilevels = multi_apply(
            self.forward_objdet_single,
            left_feature,
            right_feature,
            self._multi_level_cls_convs,
            self._multi_level_reg_convs,
            self._multi_level_conv_cls,
            self._multi_level_conv_reg,
            self._multi_level_conv_obj
        )
        # print("forward_objdet_single executed in {:.4f} seconds".format(time.time() - starttime))

        cls_scores, bboxes, objectnesses = preds_items_multilevels
        (
            loss_dict_bboxdet,
            batch_selected_boxes,
            batch_selected_classes,
            batch_selected_confidences
        ) = self.loss_by_bboxdet(
            cls_scores,
            bboxes,
            objectnesses,
            gt_labels,
            batch_img_metas
        )
        # # ======== For debug left side ========
        # num_imgs = disparity_prior.shape[0]
        # batch_positive_detections = []
        # for indexImg in range(num_imgs):
        #     batch_positive_detections.append({
        #         'bboxes': batch_selected_boxes[indexImg],
        #         'classes': batch_selected_classes[indexImg],
        #         'confidences': batch_selected_confidences[indexImg]
        #     })

        # select top candidates for stereo bbox regression.
        preds_items_multilevels_detachcopy = DetachCopyNested(preds_items_multilevels)
        (
            cls_scores_selected,  # shape [B, 100, num_class]
            bboxes_selected,  # shape is [B, 100, 4]. [tl_x, tl_y, br_x, br_y] format bbox, all in global scale.
            objectness_selected,  # Note: shape is [B, 100,].
            priors_selected  # shape [B, 100, 4]
        ) = self.SelectTopkCandidates(*preds_items_multilevels_detachcopy, img_metas=batch_img_metas)

        # return shape [B, 100, 6]
        starttime = time.time()
        sbboxes = self.forward_stereo_det(
            left_feature,
            right_feature,
            bboxes_selected,
            disparity_prior,
            batch_img_metas,
            self._config['bbox_expand_factor']
        )

        # return shape [B, 100, num_classes, 2]
        starttime = time.time()
        keypt1_pred = self.forward_keypt_det(
            left_feature,
            copy.deepcopy(bboxes_selected),
            self.keypt1_predictor
        )
        keypt2_pred = self.forward_keypt_det(
            left_feature,
            copy.deepcopy(bboxes_selected),
            self.keypt2_predictor
        )

        loss_dict_final = None
        batch_positive_detections = []
        if gt_labels is not None:
            # get loss path
            starttime = time.time()
            loss_dict_final, pos_masks = self.loss_by_stereobboxdet(
                priors_selected,
                cls_scores_selected,
                sbboxes,
                objectness_selected,
                keypt1_pred,
                keypt2_pred,
                gt_labels,
                batch_img_metas
            )
            if loss_dict_bboxdet is not None:
                loss_dict_final['loss_cls'] = loss_dict_bboxdet['loss_cls']
                loss_dict_final['loss_bbox'] = loss_dict_bboxdet['loss_bbox']
                loss_dict_final['loss_obj'] = loss_dict_bboxdet['loss_obj']

            num_imgs = disparity_prior.shape[0]
            pos_masks = pos_masks.reshape(num_imgs, -1)
            for indexImg in range(num_imgs):
                classes = torch.argmax(cls_scores_selected[indexImg][pos_masks[indexImg]].detach(), dim=-1)
                keypt1s = torch.gather(keypt1_pred[indexImg][pos_masks[indexImg]].detach(), 1, classes.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 2)).squeeze()
                keypt2s = torch.gather(keypt2_pred[indexImg][pos_masks[indexImg]].detach(), 1, classes.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 2)).squeeze()
                batch_positive_detections.append({
                    'sbboxes': sbboxes[indexImg][pos_masks[indexImg]],
                    'classes': classes,
                    'confidences': torch.max(cls_scores_selected[indexImg][pos_masks[indexImg]].detach() , dim=-1)[0] * objectness_selected[indexImg][pos_masks[indexImg]].detach(),
                    'keypt1s': keypt1s,
                    'keypt2s': keypt2s
                })

        return batch_positive_detections, loss_dict_final
    
    @torch.no_grad()
    def SelectTopkCandidates(
        self,
        cls_scores: Tuple[Tensor],
        bbox_preds: Tuple[Tensor],
        objectnesses: Tuple[Tensor],
        img_metas: Dict
    ):
        """
        select topk candidates for a batch data.
        """
        batch_size = cls_scores[0].shape[0]
        bboxes_selected, cls_scores_selected, objectness_selected, priors_selected = [], [], [], []
        for indexD in range(batch_size):
            cls_scores_one = [cls_score[indexD] for cls_score in cls_scores]
            bbox_preds_one = [bbox_pred[indexD] for bbox_pred in bbox_preds]
            objectness_one = [objectness[indexD] for objectness in objectnesses]
            cls_scores_one, bboxes_one, objectness_one, priors_one = self.SelectTopkCandidates_single(
                cls_scores_one,
                bbox_preds_one,
                objectness_one,
                img_metas=img_metas
            )
            bboxes_selected.append(bboxes_one.unsqueeze(0))
            cls_scores_selected.append(cls_scores_one.unsqueeze(0))
            objectness_selected.append(objectness_one.squeeze(-1).unsqueeze(0))
            priors_selected.append(priors_one.unsqueeze(0))
        return torch.cat(cls_scores_selected, dim=0), torch.cat(bboxes_selected, dim=0), torch.cat(objectness_selected, dim=0), torch.cat(priors_selected, dim=0)

    @torch.no_grad()
    def SelectTopkCandidates_single(
        self,
        cls_scores: Tuple[Tensor],
        bbox_preds: Tuple[Tensor],
        objectnesses: Tuple[Tensor],
        img_metas: Dict
    ):
        """
        select topk candidates based on class scores.

        Args:
            cls_scores: list contains multi-level preds result. it does not contain batch dimension.
            bbox_preds: list contains multi-level preds result.
            objectnesses: list contains multi-level preds result.
        """
        num_imgs = cls_scores[0].shape[0]
        featmap_sizes = [cls_score.shape[-2:] for cls_score in cls_scores]
        # Follow YOLOX design:
        # uses center priors with 0.5 offset to assign targets,
        # but use center priors without offset to regress bboxes.
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device,
            with_stride=True
        )

        img_shape = (img_metas['h'], img_metas['w'])
        nms_pre = self._config.get('nms_pre', -1)
        mlvl_bbox_preds = []
        mlvl_valid_priors = []
        mlvl_confidences = []
        mlvl_cls_scores = []
        mlvl_objectness = []
        level_ids = []
        for level_idx, (cls_score, bbox_pred, objectness, priors) in enumerate(zip(cls_scores, bbox_preds, objectnesses, mlvl_priors)):
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            cls_score = cls_score.permute(1, 2, 0).reshape(-1, self._config['num_classes']).sigmoid()
            objectness = objectness.permute(1, 2, 0).reshape(-1, 1).sigmoid()
            max_scores, labels = torch.max(cls_score, -1)
            confidences = max_scores * objectness.squeeze(-1)
            if 0 < nms_pre < confidences.shape[0]:
                ranked_confidences, rank_inds = confidences.sort(descending=True)
                topk_inds = rank_inds[:nms_pre]
                confidences = ranked_confidences[:nms_pre]
                bbox_pred = bbox_pred[topk_inds, :]
                priors = priors[topk_inds]
                cls_score = cls_score[topk_inds]
                objectness = objectness[topk_inds]
            mlvl_bbox_preds.append(bbox_pred)
            mlvl_valid_priors.append(priors)
            mlvl_confidences.append(confidences)
            mlvl_cls_scores.append(cls_score)
            mlvl_objectness.append(objectness)
            # use level id to implement the separate level nms
            level_ids.append(
                confidences.new_full((confidences.size(0), ),
                                level_idx,
                                dtype=torch.long))
        bbox_pred = torch.cat(mlvl_bbox_preds)
        priors = cat_boxes(mlvl_valid_priors)
        bboxes = self._bbox_decode(priors, bbox_pred)

        results = InstanceData()
        results.bboxes = bboxes
        results.scores = torch.cat(mlvl_confidences)
        results.cls_scores = torch.cat(mlvl_cls_scores)
        results.priors = priors
        results.objectness = torch.cat(mlvl_objectness)
        results.level_ids = torch.cat(level_ids)
        # filter small size bboxes
        if self._config.get('min_bbox_size', -1) >= 0:
            w = results.bboxes[:, 2] - results.bboxes[:, 0]
            h = results.bboxes[:, 3] - results.bboxes[:, 1]
            valid_mask = (w > self._config['min_bbox_size']) & (h > self._config['min_bbox_size'])
            if not valid_mask.all():
                results = results[valid_mask]

        det_bboxes, keep_idxs = batched_nms(
            results.bboxes,
            results.scores,
            results.level_ids,
            {'type': 'nms', 'iou_threshold': 0.7}
        )
        results = results[keep_idxs]
        if results.bboxes.shape[0] < self._config['num_topk_candidates']:
            # patching missing length with zeros
            len_missing = self._config['num_topk_candidates'] - results.bboxes.shape[0]
            bboxes_toadd = results.bboxes.new_zeros((len_missing, 4))
            scores_toadd = results.scores.new_zeros((len_missing,))
            priors_toadd = results.priors.new_zeros((len_missing, results.priors.shape[-1]))
            cls_scores_toadd = results.cls_scores.new_zeros((len_missing, results.cls_scores.shape[-1]))
            objectness_toadd = results.objectness.new_zeros((len_missing, 1))
            results.bboxes = torch.cat((results.bboxes, bboxes_toadd))
            results.scores = torch.cat((results.scores, scores_toadd))
            results.priors = torch.cat((results.priors, priors_toadd))
            results.cls_scores = torch.cat((results.cls_scores, cls_scores_toadd))
            results.objectness = torch.cat((results.objectness, objectness_toadd))
        else:
            results = results[:self._config['num_topk_candidates']]
        return results.cls_scores, results.bboxes, results.objectness, results.priors

    def forward_objdet_single(
        self,
        left_feat: Tensor,
        right_feat: Tensor,
        cls_convs: nn.Module,
        reg_convs: nn.Module,
        conv_cls: nn.Module,
        conv_reg: nn.Module,
        conv_obj: nn.Module,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        forward feature of a single scale level
        Args:
            left_feat: (b, c, h, w) tensor from left camera of stereo pair.
            ...
        
        Returns:
            cls_score (Tensor): (b, num_classes, h, w) shape. 
                                probabilities of class of the bbox at each grid.
            bbox_pred (Tensor): (b, 4, h, w) shape.
                                each grid contains one predicted bbox. 
                                The dimension 4 contains [delta_x, delta_y, w_l, h_l].
                                Note that stereo disparity will not be estimated here. 
                                A rough disp estimation is from external, then fine disp will be estimated in stereo bbox regression part.
            objectness (Tensor): (b, 1, h, w) shape.
                                 Confidence of whether the object in the bbox is an object.
                                 Will be used for non-maximum suppression.
        """
        # x = torch.cat((left_feat, right_feat), 1)
        x = left_feat
        # cls, bbox, objectness prediction, similar to yolo
        cls_feat = cls_convs(x)
        reg_feat = reg_convs(x)

        cls_score = conv_cls(
            cls_feat
        )
        bbox_pred = conv_reg(
            reg_feat
        )
        objectness = conv_obj(
            reg_feat
        )
        return cls_score, bbox_pred, objectness

    def forward_stereo_det(
        self,
        left_feat: List[Tensor],
        right_feat: List[Tensor],
        bboxes_pred: Tensor,
        disp_prior: Tensor,
        batch_img_metas: Dict,
        bbox_expand_factor: float
    ) -> Tensor:
        """
        Args:
            left_feat: multi level features from left.
            right_feat: multi level features from right.
            bboxes_pred: shape is [B, 100, 4]. [tl_x, tl_y, br_x, br_y] format bbox, all in global scale.
            disp_prior: [B, h, w] shape.
            bbox_expand_factor: ratio to enlarge bbox due to inaccurate disp prior.

        Returns:
            sbboxes_pred: shape [B, 100, 6]. format [tl_x, tl_y, br_x, br_y, tl_x_r, br_x_r] stereo bbox
        """
        if self.logger is not None:
            right_feat_sample = right_feat[0][0, 0, :, :].detach().cpu()
            right_feat_sample = right_feat_sample - right_feat_sample.min()
            right_feat_sample /= right_feat_sample.max()
            right_feat_sample = F.interpolate(right_feat_sample.unsqueeze(0).unsqueeze(0), size=(720, 1280), mode='bilinear', align_corners=False)
            self.logger.add_image("right_feat_sample", right_feat_sample.squeeze())

        # enlarge bboxes
        _bboxes_pred = bboxes_pred.clone()  # initial warped bboxes for right side target.
        dwboxes = (_bboxes_pred[..., 2] - _bboxes_pred[..., 0]) * (bbox_expand_factor - 1.)
        # dhboxes = (_bboxes_pred[..., 3] - _bboxes_pred[..., 1]) * (bbox_expand_factor - 1.)
        _bboxes_pred[..., 0] -= dwboxes / 2  # Note: only need to enlarge x direction, because we assume for stereo y is aligned.
        _bboxes_pred[..., 2] += dwboxes / 2
        # _bboxes_pred[..., 1] -= dhboxes / 2
        # _bboxes_pred[..., 3] += dhboxes / 2

        batch_number = torch.arange(_bboxes_pred.shape[0]).unsqueeze(1).expand(-1, self._config['num_topk_candidates']).flatten().unsqueeze(-1).to(_bboxes_pred.device)

        # extract right bbox roi feature
        xindi = ((_bboxes_pred[..., 0] + _bboxes_pred[..., 2]) / 2).to(torch.int).clamp(0, batch_img_metas['w'] - 1)
        yindi = ((_bboxes_pred[..., 1] + _bboxes_pred[..., 3]) / 2).to(torch.int).clamp(0, batch_img_metas['h'] - 1)
        bbox_disps = disp_prior[:, yindi, xindi].squeeze(0)  # Note: dimension will increase one due to xindi, yindi shape
        _bboxes_pred[..., 0] -= bbox_disps  # warp to more left side.
        _bboxes_pred[..., 2] -= bbox_disps
        rois_right = _bboxes_pred.reshape(-1, 4)
        rois_right = torch.cat((batch_number, rois_right), dim=1)
        right_roi_feats = self.bbox_roi_extractor(right_feat, rois_right)  # Note: Based on the bbox size to decide from which level to extract feats.
        right_bboxes_pred = self.right_bbox_refiner(right_roi_feats)
        if self.logger is not None:
            roi_feat_sample = right_roi_feats[0, 0, :, :].detach().cpu()
            roi_feat_sample = roi_feat_sample - roi_feat_sample.min()
            roi_feat_sample /= roi_feat_sample.max()
            self.logger.add_image("roi_feat_sample", roi_feat_sample)

        xc_r = (_bboxes_pred[..., 2] - _bboxes_pred[..., 0]) * right_bboxes_pred[:, 0] + _bboxes_pred[..., 0]
        w_r = right_bboxes_pred[:, 1] * (_bboxes_pred[..., 2] - _bboxes_pred[..., 0])
        x_tl_r = (xc_r - w_r / 2).view(bboxes_pred.shape[0], -1).unsqueeze(-1)
        x_br_r = (xc_r + w_r / 2).view(bboxes_pred.shape[0], -1).unsqueeze(-1)
        sbboxes_pred = torch.cat([bboxes_pred, x_tl_r, x_br_r], dim=-1)
        return sbboxes_pred

    def forward_keypt_det(
        self,
        left_feat: List[Tensor],
        bbox_pred: Tensor,
        keypt_predictor: nn.Sequential
    ) -> Tensor:
        """
        bbox_pred is top 100 bbox predictions from forward_objdet_single. Same as mask-rcnn.
        predict keypoint bias to the grid center, e.g. [delta_x, delta_y].
        predict one keypoint for each class, shape of output is (b, 100, num_classes, 2). only take the k-th keypoint, k is the index of class of the bbox.
        Args:
            left_feat: multi level features from left.
            bbox_pred: (b, 100, 4) tensor. predicted bbox of objects. 100 is the number of bboxes. 4 channels are [tl_x, tl_y, br_x, br_y] format bbox.

        Return:
            keypts_pred: (b, 100, num_classes, 2) tensor. Predicted keypoint bias (normalized) to the bbox top left corner.
        """
        batch_number = torch.arange(bbox_pred.shape[0]).unsqueeze(1).expand(-1, self._config['num_topk_candidates']).flatten().unsqueeze(-1).to(bbox_pred.device)
        rois = bbox_pred.reshape(-1, 4)
        bnum_rois = torch.cat([batch_number, rois], dim=1)
        roi_feats = self.bbox_roi_extractor(left_feat, bnum_rois)  # Note: output shape is (b*100, 128, 14, 14)
        keypts_feat = keypt_predictor(roi_feats)  # Note: (b*100, num_of_class, 28, 28)
        # FIXME: not using torch.argmax, instead multiply prob distribution to a torch.arange to get the coordinates in a differentiable way. 
        map_size = keypts_feat.shape[-1]
        keypts_prob_x = ComputeSoftArgMax1d(keypts_feat.sum(dim=-2))  # Note: shape is (b*100, num_classes, 1)
        keypts_prob_x = keypts_prob_x.unsqueeze(-1)
        keypts_prob_y = ComputeSoftArgMax1d(keypts_feat.sum(dim=-1))
        keypts_prob_y = keypts_prob_y.unsqueeze(-1)
        keypts_pred = torch.cat(
            (keypts_prob_x, keypts_prob_y), dim=-1
        ) / map_size  # Note: shape is (b*100, num_classes, 2)
        batch_size, num_bboxes = bbox_pred.shape[:2]
        num_classes, num_pts_dimension = keypts_pred.shape[-2:]
        return keypts_pred.view(batch_size, num_bboxes, num_classes, num_pts_dimension)

    def loss_by_stereobboxdet(
        self,
        priors: Tensor,
        cls_scores: Tensor,
        bboxes: Tensor,
        objectness: Tensor,
        keypt1_pred: Tensor,
        keypt2_pred: Tensor,
        batch_gt_labels: Dict,
        batch_img_metas: Dict
    ):
        """
        Compute losses for object detection.

        Args:
            priors: shape [B, 100, 4],
            cls_scores: shape [B, 100, num_class]
            bboxes: shape [B, 100, 6]. [tl_x, tl_y, br_x, br_y, tl_x_r, br_x_r] format bbox, all in global scale.
            objectness: shape [B, 100].
            keypt1_pred: [B, 100, num_class, 2]
            keypt2_pred: [B, 100, num_class, 2]
            batch_gt_labels: include 'bboxes' and 'labels' keys.
            batch_img_metas: include 'h' and 'w' keys. 

        Returns:
            loss_dict: ...
            pos_masks: positive mask for the batch data.
        """
        num_imgs = priors.shape[0]

        (
            pos_masks,
            cls_targets,
            obj_targets,  # If it is a thing, no matter what class, objectness_target is 1.0
            bbox_targets,
            keypt1_targets,
            keypt2_targets,
            num_pos_per_img  # number of positive gt target in each image.
        ) = multi_apply(
            self._get_targets_stereo_single,
            priors.detach(),
            cls_scores.detach(),
            bboxes.detach(),
            objectness.detach(),
            keypt1_pred.detach(),
            keypt2_pred.detach(),
            batch_gt_labels,
            batch_img_metas
        )

        num_pos = torch.tensor(
            sum(num_pos_per_img),
            dtype=torch.float,
            device=cls_scores.device
        )
        num_total_samples = max(reduce_mean(num_pos), 1.0)

        pos_masks = torch.cat(pos_masks, 0)
        cls_targets = torch.cat(cls_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        bbox_targets = torch.cat(bbox_targets, 0)
        keypt1_targets = torch.cat(keypt1_targets, 0)
        keypt2_targets = torch.cat(keypt2_targets, 0)

        if num_pos > 0:
            rbboxes = torch.cat([
                bboxes.view(-1, 6)[pos_masks][:, 4].unsqueeze(-1),
                bboxes.view(-1, 6)[pos_masks][:, 1].unsqueeze(-1).detach(),
                bboxes.view(-1, 6)[pos_masks][:, 5].unsqueeze(-1),
                bboxes.view(-1, 6)[pos_masks][:, 3].unsqueeze(-1).detach()
            ], dim=1)
            rbboxes_targets = torch.cat([
                bbox_targets[:, 4].unsqueeze(-1),
                bbox_targets[:, 1].unsqueeze(-1),
                bbox_targets[:, 5].unsqueeze(-1),
                bbox_targets[:, 3].unsqueeze(-1)
            ], dim=1)
            loss_rbbox = self.loss_rbbox(rbboxes, rbboxes_targets) / num_total_samples
            # keypoints
            classSelection = torch.argmax(cls_scores, dim=2).reshape(-1)
            loss_keypts = []        
            for keypt_pred, losser_keypt, keypt_targets in zip([keypt1_pred, keypt2_pred], [self.loss_keypt1, self.loss_keypt2], [keypt1_targets, keypt2_targets]):
                num_batch, num_priors = keypt_pred.shape[:2]
                class_selected_keypt_pred = torch.gather(keypt_pred.reshape(-1, self._config['num_classes'], 2), 1, classSelection.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 2)).squeeze()
                loss_keypts.append(losser_keypt(class_selected_keypt_pred[pos_masks], keypt_targets) / num_total_samples)
        else:
            # if no gt, then not update
            # see https://github.com/open-mmlab/mmdetection/issues/7298
            loss_keypts = [
                keypt1_pred.sum() * 0,
                keypt2_pred.sum() * 0
            ]
            loss_rbbox = bboxes[..., 4:].sum() * 0
        loss_dict = {
            'loss_rbbox': loss_rbbox,
            'loss_keypt1': loss_keypts[0],
            'loss_keypt2': loss_keypts[1]
        }
        return  loss_dict, pos_masks.reshape(num_imgs, -1)

    def _sbbox_decode(self, flatten_priors: Tensor, flatten_sbbox_preds: Tensor) -> Tensor:
        """
        Decode stereo bbox regression result (delta_x, delta_y, w, h, delta_x_r, w_r) to
        sbboxes (tl_x, tl_y, br_x, br_y, tl_x_r, br_x_r) format.
        Args:
            ...
        """
        xys = (flatten_sbbox_preds[..., :2] * flatten_priors[:, 2:]) + flatten_priors[:, :2]
        whs = flatten_sbbox_preds[..., 2:4].exp() * flatten_priors[:, 2:]
        x_r = (flatten_sbbox_preds[..., 4].exp() * flatten_priors[:, 2]) + flatten_priors[:, 0]
        w_r = flatten_sbbox_preds[..., 5].exp() * flatten_priors[:, 2]

        tl_x = (xys[..., 0] - whs[..., 0] / 2)
        tl_y = (xys[..., 1] - whs[..., 1] / 2)
        br_x = (xys[..., 0] + whs[..., 0] / 2)
        br_y = (xys[..., 1] + whs[..., 1] / 2)
        tl_x_r = x_r - (w_r / 2)
        br_x_r = x_r + (w_r / 2)
        
        decoded_sbboxes = torch.stack([tl_x, tl_y, br_x, br_y, tl_x_r, br_x_r], dim=-1)
        return decoded_sbboxes

    def _bbox_decode(self, priors: Tensor, bbox_preds: Tensor) -> Tensor:
        """
        Decode bbox regression result (delta_x, delta_y, w, h, delta_x_r, w_r) to
        bboxes (tl_x, tl_y, br_x, br_y) format.

        Args:
            priors (Tensor): Center proiors of an image, has shape
                (num_instances, 2).
            bbox_preds (Tensor): Box energies / deltas for all instances,
                has shape (batch_size, num_instances, 4).

        Returns:
            Tensor: Decoded bboxes in (tl_x, tl_y, br_x, br_y) format. Has
            shape (batch_size, num_instances, 4).
        """
        xys = (bbox_preds[..., :2] * priors[:, 2:]) + priors[:, :2]
        whs = bbox_preds[..., 2:].exp() * priors[:, 2:]

        tl_x = (xys[..., 0] - whs[..., 0] / 2)
        tl_y = (xys[..., 1] - whs[..., 1] / 2)
        br_x = (xys[..., 0] + whs[..., 0] / 2)
        br_y = (xys[..., 1] + whs[..., 1] / 2)

        decoded_bboxes = torch.stack([tl_x, tl_y, br_x, br_y], -1)
        return decoded_bboxes

    @torch.no_grad()
    def _get_targets_stereo_single(
        self,
        priors: Tensor,
        cls_scores: Tensor,
        bboxes: Tensor,
        objectness: Tensor,
        keypt1_pred: Tensor,
        keypt2_pred: Tensor,
        gt_labels: Dict,
        img_metas: Dict
    ) -> tuple:
        """
        Compute classification, regression, and objectness targets for priors in the left image
    
        Args:
            priors (Tensor): Grid priors of the image. 
                             A 2D tensor with shape [num_priors, 4] in [cx, cy, stride_w, stride_y] format.
                             each grid represents one pred at that pixel.
            cls_scores (Tensor): Classification predictions of one image. 
                                 A 2D tensor with shape [num_priors, num_classes].
            bboxes (Tensor): Decoded bboxes predictions. 
                             A 2D tensor with shape [num_priors, 6].
            objectness (Tensor): Objectness predictions of one image, a 1D tensor with shape [num_priors]
            keypt1_pred (Tensor): shape [num_priors, num_class, 2].
            keypt2_pred (Tensor): shape [num_priors, num_class, 2].
            gt_labels (Dict): It includes 'bboxes' (num_instances, 10) and 'labels' (num_instances,).
            img_metas (Dict): meta info about the input image width and height. 
        """
        num_priors = priors.shape[0]
        num_gts = gt_labels['bboxes'].shape[0]

        # no target
        if num_gts == 0:
            cls_target = cls_scores.new_zeros((0, self._config['num_classes']))
            bbox_target = cls_scores.new_zeros((0, 6))
            obj_target = cls_scores.new_zeros((num_priors, 1))
            pos_mask = cls_scores.new_zeros(num_priors).bool()
            keypt1_target = cls_scores.new_zeros((0, self._config['num_classes'], 2))
            keypt2_target = cls_scores.new_zeros((0, self._config['num_classes'], 2))
            return (pos_mask, cls_target, obj_target, bbox_target, keypt1_target, keypt2_target, 0)

        # (refer to YOLOX) use center priors with 0.5 offset to assign targets,
        # but use center priors without offset to regress bboxes
        offset_priors = torch.cat([priors[:, :2] + priors[:, 2:] * 0.5, priors[:, 2:]], dim=-1)
        scores = cls_scores.sigmoid() * objectness.unsqueeze(1).sigmoid()
        pred_instances = InstanceData(
            bboxes=bboxes[:, :4],
            scores=scores.sqrt_(),
            priors=offset_priors
        )
        gt_instances = InstanceData(
            bboxes=gt_labels['bboxes'][:, :4],
            labels=gt_labels['labels'],
            sbboxes=gt_labels['bboxes'][:, :]
        )
        # use SimOTA dynamic assigner, same as yolox
        assign_result = self.assigner.assign(
            pred_instances=pred_instances,
            gt_instances=gt_instances
        )

        # use a pesudo sampler to get all results. just for using mmdet util's api.
        sampling_result = self.sampler.sample(
            assign_result,
            pred_instances,
            gt_instances
        )
        pos_inds = sampling_result.pos_inds
        num_pos_per_img = pos_inds.size(0)

        pos_ious = assign_result.max_overlaps[pos_inds]
        # Yolox: IoU aware classification scores
        cls_target = F.one_hot(sampling_result.pos_gt_labels, self._config['num_classes']) * pos_ious.unsqueeze(-1)
        obj_target = torch.zeros_like(objectness).unsqueeze(-1)
        obj_target[pos_inds] = 1
        bbox_target = sampling_result.pos_gt_sbboxes
        keypts1_target = sampling_result.pos_keypts1
        keypts2_target = sampling_result.pos_keypts2
        pos_mask = torch.zeros_like(objectness).to(torch.bool)
        pos_mask[pos_inds] = True
        return (pos_mask, cls_target, obj_target, bbox_target, keypts1_target, keypts2_target, num_pos_per_img)

    def loss_by_bboxdet(
        self,
        cls_scores: Tuple[Tensor],
        bbox_preds: Tuple[Tensor],
        objectnesses: Tuple[Tensor],
        gt_labels: Dict,
        batch_img_metas: Dict
    ) -> Tuple:
        """
        Train left side object detection only.
        """
        num_imgs = len(gt_labels)
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device,
            with_stride=True)

        flatten_cls_preds = [
            cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                 self._config['num_classes'])
            for cls_pred in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_objectness = [
            objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
            for objectness in objectnesses
        ]

        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_objectness = torch.cat(flatten_objectness, dim=1)
        flatten_priors = torch.cat(mlvl_priors)
        flatten_bboxes = self._bbox_decode(flatten_priors, flatten_bbox_preds)

        (
            pos_masks,
            cls_targets,
            obj_targets,
            bbox_targets,
            num_pos_imgs
        ) = multi_apply(
            self._get_targets_single,
            flatten_priors.unsqueeze(0).repeat(num_imgs, 1, 1),
            flatten_cls_preds.detach(),
            flatten_bboxes.detach(),
            flatten_objectness.detach(),
            gt_labels,
            batch_img_metas)

        num_pos = torch.tensor(
            sum(num_pos_imgs),
            dtype=torch.float,
            device=flatten_cls_preds.device)
        num_total_samples = max(reduce_mean(num_pos), 1.0)

        pos_masks = torch.cat(pos_masks, 0)
        cls_targets = torch.cat(cls_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        bbox_targets = torch.cat(bbox_targets, 0)

        loss_obj = self.loss_obj(flatten_objectness.view(-1, 1), obj_targets) / num_total_samples
        if num_pos > 0:
            loss_cls = self.loss_cls(
                flatten_cls_preds.view(-1, self._config['num_classes'])[pos_masks],
                cls_targets) / num_total_samples
            loss_bbox = self.loss_bbox(
                flatten_bboxes.view(-1, 4)[pos_masks],
                bbox_targets) / num_total_samples
        else:
            # Avoid cls and reg branch not participating in the gradient
            # propagation when there is no ground-truth in the images.
            # For more details, please refer to
            # https://github.com/open-mmlab/mmdetection/issues/7298
            loss_cls = flatten_cls_preds.sum() * 0
            loss_bbox = flatten_bboxes.sum() * 0
        loss_dict = {
            'loss_cls': loss_cls,
            'loss_bbox': loss_bbox,
            'loss_obj': loss_obj
        }

        batch_selected_boxes = []
        batch_selected_classes = []
        batch_selected_confidences = []
        pos_masks = pos_masks.reshape(num_imgs, -1)
        for index_img in range(num_imgs):
            batch_selected_boxes.append(
                flatten_bboxes[index_img][pos_masks[index_img]].detach()
            )
            batch_selected_classes.append(
                flatten_cls_preds[index_img][pos_masks[index_img]].detach()
            )
            batch_selected_confidences.append(
                torch.max(flatten_cls_preds[index_img][pos_masks[index_img]], dim=-1)[0].detach() * flatten_objectness[index_img][pos_masks[index_img]].detach()
            )
        return  loss_dict, batch_selected_boxes, batch_selected_classes, batch_selected_confidences

    def _get_targets_single(
        self,
        priors: Tensor,
        cls_preds: Tensor,
        decoded_bboxes: Tensor,
        objectness: Tensor,
        gt_labels: InstanceData,
        img_meta: dict,
    ):
        num_priors = priors.shape[0]
        num_gts = gt_labels['bboxes'].shape[0]
        # No target
        if num_gts == 0:
            cls_target = cls_preds.new_zeros((0, self.num_classes))
            bbox_target = cls_preds.new_zeros((0, 4))
            obj_target = cls_preds.new_zeros((num_priors, 1))
            pos_mask = cls_preds.new_zeros(num_priors).bool()
            return (pos_mask, cls_target, obj_target, bbox_target, 0)

        # YOLOX uses center priors with 0.5 offset to assign targets,
        # but use center priors without offset to regress bboxes.
        offset_priors = torch.cat(
            [priors[:, :2] + priors[:, 2:] * 0.5, priors[:, 2:]], dim=-1)

        scores = cls_preds.sigmoid() * objectness.unsqueeze(1).sigmoid()
        pred_instances = InstanceData(
            bboxes=decoded_bboxes,
            scores=scores.sqrt_(),
            priors=offset_priors
        )
        gt_instances = InstanceData(
            bboxes=gt_labels['bboxes'][:, :4],
            labels=gt_labels['labels']
        )
        assign_result = self.assigner.assign(
            pred_instances=pred_instances,
            gt_instances=gt_instances
        )
        sampling_result = self.sampler.sample(assign_result, pred_instances, gt_instances)

        pos_inds = sampling_result.pos_inds
        num_pos_per_img = pos_inds.size(0)

        pos_ious = assign_result.max_overlaps[pos_inds]
        # IOU aware classification score
        cls_target = F.one_hot(sampling_result.pos_gt_labels,
                               self._config['num_classes']) * pos_ious.unsqueeze(-1)
        obj_target = torch.zeros_like(objectness).unsqueeze(-1)
        obj_target[pos_inds] = 1
        bbox_target = sampling_result.pos_gt_bboxes
        pos_mask = torch.zeros_like(objectness).to(torch.bool)
        pos_mask[pos_inds] = 1
        return (pos_mask, cls_target, obj_target, bbox_target, num_pos_per_img)


