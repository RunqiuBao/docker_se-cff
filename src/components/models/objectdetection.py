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
from mmengine.structures import InstanceData
from mmdet.structures.bbox import cat_boxes
from mmdet.registry import TASK_UTILS, MODELS
from mmdet.structures.mask import mask_target, BitmapMasks
from mmengine.config import Config
from mmcv.ops import batched_nms

from .utils.misc import DetachCopyNested, freeze_module_grads
from . import losses


class Cylinder5DDetectionHead(nn.Module):

    def __init__(self, net_cfg: Dict, loss_cfg: Dict, is_distributed: bool, logger=None):
        super().__init__()
        self._config = net_cfg
        self.logger = logger
        self.strides = [8, 16, 32]
        self.keypts_roi_visz = {
            '1': None,
            '2': None
        }
        self.keypts_featmap_visz = {
            '1': None,
            '2': None
        }

        self._init_layers()
        self._init_weights()

        if self._config['is_train_keypt']:
            freeze_module_grads(self._multi_level_cls_convs)
            freeze_module_grads(self._multi_level_reg_convs)
            freeze_module_grads(self._multi_level_conv_cls)
            freeze_module_grads(self._multi_level_conv_reg)
            freeze_module_grads(self._multi_level_conv_obj)
            freeze_module_grads(self.right_bbox_refiner)
            freeze_module_grads(self.right_bbox_refiner_scorer)

        # objdet tools
        self.assigner = TASK_UTILS.build({'type': 'SimOTAAssigner', 'center_radius': 2.5})
        self.sampler = PseudoSampler()
        self.bbox_roi_extractor = MODELS.build(
            {
                'type': 'SingleRoIExtractor',
                'roi_layer': {'type': 'RoIAlign', 'output_size': self._config['right_roi_feat_size'], 'sampling_ratio': 0},
                'out_channels': self._config['in_channels'],
                'featmap_strides': self.strides
            }
        )
        self.keypts_bbox_roi_extractor = MODELS.build(
            {
                'type': 'SingleRoIExtractor',
                'roi_layer': {'type': 'RoIAlign', 'output_size': self._config['keypts_feat_size'], 'sampling_ratio': 0},
                'out_channels': self._config['keypt_in_channels'],
                'featmap_strides': [1]
            }
        )
        self.iou_calculator = TASK_UTILS.build({'type': 'BboxOverlaps2D'})
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
                                        'loss_weight': 5.0})
        self.loss_rscore = MODELS.build({'type': 'CrossEntropyLoss',
                                      'use_sigmoid': True,
                                      'reduction': 'sum',
                                      'loss_weight': 1.0})
        # Need to consider keypts in loss_bbox as well.
        self.loss_keypt1 = torch.nn.SmoothL1Loss(reduction='mean') #MODELS.build({'type': 'CrossEntropyLoss', 'use_mask': True, 'loss_weight': 1.0})
        self.loss_keypt2 = torch.nn.SmoothL1Loss(reduction='mean') #MODELS.build({'type': 'CrossEntropyLoss', 'use_mask': True, 'loss_weight': 1.0})
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

        # Note: two keypoints needed. One at object center, one at object top center.
        self.keypt1_predictor = self._build_keypts_convs(
            in_channels=self._config["keypt_in_channels"],
            feat_channels=self._config["keypt_feat_channels"],
            num_classes=self._config["num_classes"],
        )
        self.keypt2_predictor = self._build_keypts_convs(
            in_channels=self._config["keypt_in_channels"],
            feat_channels=self._config["keypt_feat_channels"],
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

    def _init_weights(self):
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

        for subnet in [self.right_bbox_refiner]:
            for module in subnet.modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    nn.init.kaiming_uniform_(
                        module.weight,
                        a=math.sqrt(5),
                        mode="fan_in",
                        nonlinearity="leaky_relu"
                    )
        for subnet in [self.keypt1_predictor, self.keypt2_predictor]:
            for m in subnet.modules():
                if m is None:
                    continue
                elif hasattr(m, 'weight') and hasattr(m, 'bias'):
                    nn.init.kaiming_normal_(
                        m.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.constant_(m.bias, 0)

        prior_prob = 0.01
        bias_init = float(-numpy.log((1 - prior_prob) / prior_prob))
        for module_list in [self._multi_level_conv_cls, self._multi_level_conv_obj]:
            for module in module_list:
                module.bias.data.fill_(bias_init)
        
        # self.right_bbox_refiner_scorer_output.bias.data.fill_(bias_init)

    @staticmethod
    def _build_keypts_convs(in_channels: int, feat_channels: int, num_classes: int):
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

    def forward(
        self,
        left_feature: List[Tensor],
        right_feature: List[Tensor],
        keypt_left_feat: List[Tensor],
        disparity_prior: Tensor,
        batch_img_metas: Dict,
        gt_labels: List[Dict] = None,
    ):
        """
        Args:
            left_feature: multi-level features from left image input.
            right_feature: multi-level features from right image input.
            gt_labels: list of dict for one batch. each dict contains 'bboxes' and 'labels' keys.
        Returns:
            Tuple[List]: A tuple of multi-level classification scores, bbox predictions, keypts predcitions, objectnesses.
        """
        cls_scores, bboxes, objectnesses = [], [], []
        for indexLevel in range(len(left_feature)):
            cls_scores_one, bboxes_one, objectnesses_one = self.forward_objdet_single(
                left_feature[indexLevel],
                right_feature[indexLevel],
                self._multi_level_cls_convs[indexLevel],
                self._multi_level_reg_convs[indexLevel],
                self._multi_level_conv_cls[indexLevel],
                self._multi_level_conv_reg[indexLevel],
                self._multi_level_conv_obj[indexLevel]
            )
            cls_scores.append(cls_scores_one)
            bboxes.append(bboxes_one)
            objectnesses.append(objectnesses_one)
        torch.cuda.synchronize()

        if gt_labels is not None:
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
            torch.cuda.synchronize()

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
        preds_items_multilevels_detachcopy = DetachCopyNested([cls_scores, bboxes, objectnesses])
        (
            cls_scores_selected,  # shape [B, 100, num_class]
            bboxes_selected,  # shape is [B, 100, 4]. [tl_x, tl_y, br_x, br_y] format bbox, all in global scale.
            objectness_selected,  # Note: shape is [B, 100,].
            priors_selected  # shape [B, 100, 4]
        ) = self.SelectTopkCandidates(*preds_items_multilevels_detachcopy, img_metas=batch_img_metas)        

        # return shape [B, 100, 6]
        sbboxes, refined_right_bboxes, refined_right_scores = self.forward_stereo_det(
            right_feature,
            bboxes_selected,
            disparity_prior,
            batch_img_metas,
            self._config['bbox_expand_factor']
        )
        torch.cuda.synchronize()

        keypt1_pred = self.forward_keypt_det(
            keypt_left_feat,
            bboxes_selected.detach(),
            self.keypt1_predictor,
            keypt_id='1'
        )
        keypt2_pred = self.forward_keypt_det(
            keypt_left_feat,
            bboxes_selected.detach(),
            self.keypt2_predictor,
            keypt_id='2',
        )
        torch.cuda.synchronize()

        loss_dict_final = None
        batch_positive_detections = []
        if gt_labels is not None:
            # get loss path
            loss_dict_final, pos_masks, sbboxes = self.loss_by_stereobboxdet(
                priors_selected,
                cls_scores_selected,
                sbboxes,
                refined_right_bboxes,
                refined_right_scores,
                objectness_selected,
                keypt1_pred,
                keypt2_pred,
                gt_labels,
                batch_img_metas
            )
            torch.cuda.synchronize()

            if loss_dict_bboxdet is not None:
                loss_dict_final['loss_cls'] = loss_dict_bboxdet['loss_cls']
                loss_dict_final['loss_bbox'] = loss_dict_bboxdet['loss_bbox']
                loss_dict_final['loss_obj'] = loss_dict_bboxdet['loss_obj']

            num_imgs = disparity_prior.shape[0]
            pos_masks = pos_masks.reshape(num_imgs, -1)
            if pos_masks.sum() > 0:
                for indexImg in range(num_imgs):
                    classes = torch.argmax(cls_scores_selected[indexImg][pos_masks[indexImg]].detach(), dim=-1)
                    featmap_size = keypt1_pred.shape[-1]
                    keypt1s = torch.gather(keypt1_pred[indexImg][pos_masks[indexImg]].detach(), 1, classes.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, featmap_size, featmap_size)).squeeze()
                    keypt2s = torch.gather(keypt2_pred[indexImg][pos_masks[indexImg]].detach(), 1, classes.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, featmap_size, featmap_size)).squeeze()
                    num_pos = keypt1s.shape[0]
                    indices_keypt1s = torch.argmax(keypt1s.view(num_pos, -1), dim=-1)
                    indices_keypt1s = torch.stack([indices_keypt1s % featmap_size, indices_keypt1s // featmap_size], dim=-1)
                    indices_keypt2s = torch.argmax(keypt2s.view(num_pos, -1), dim=-1)
                    indices_keypt2s = torch.stack([indices_keypt2s % featmap_size, indices_keypt2s // featmap_size], dim=-1)

                    batch_positive_detections.append({
                        'sbboxes': sbboxes[indexImg][pos_masks[indexImg]],
                        'classes': classes,
                        'confidences': torch.max(cls_scores_selected[indexImg][pos_masks[indexImg]].detach() , dim=-1)[0] * objectness_selected[indexImg][pos_masks[indexImg]].detach(),
                        'keypt1s': indices_keypt1s.to(torch.float) / featmap_size,
                        'keypt2s': indices_keypt2s.to(torch.float) / featmap_size
                    })

            if self._config['is_train_keypt']:
                active_losses = ["loss_keypt1", "loss_keypt2"]
                for key, value in loss_dict_final.items():
                    if key not in active_losses:
                        loss_dict_final[key] *= 0
        else:
            # inference code
            batch_positive_detections = self.FormatPredictionResult(
                bboxes_selected,
                cls_scores_selected,
                objectness_selected,
                refined_right_bboxes,
                refined_right_scores,
                keypt1_pred,
                keypt2_pred,
                batch_img_metas
            )

        torch.distributed.barrier()
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
            {'type': 'nms', 'iou_threshold': self._config['nms_iou_threshold']}
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
            sbboxes_pred: shape [B, 100, 6]. format [tl_x, tl_y, br_x, br_y, tl_x_r, br_x_r] rough stereo bbox
            refined_right_bboxes: shape [B, 100, ker_h * ker_w, 4]. Corresponding refined right bboxes.
            right_scores_refine: shape [B, 100, ker_h * ker_w, 1]. Corresponding scores.
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
        bbox_disps = []
        for indexInBatch, disp_prior_one in enumerate(disp_prior):
            bbox_disps.append(disp_prior_one[yindi[indexInBatch], xindi[indexInBatch]].unsqueeze(0))  # Note: dimension will increase one due to xindi, yindi shape
        bbox_disps = torch.cat(bbox_disps, dim=0)
        _bboxes_pred[..., 0] -= bbox_disps  # warp to more left side.
        _bboxes_pred[..., 2] -= bbox_disps
        rois_right = _bboxes_pred.reshape(-1, 4)
        rois_right = torch.cat((batch_number, rois_right), dim=1)
        right_roi_feats = self.bbox_roi_extractor(right_feat, rois_right)  # Note: Based on the bbox size to decide from which level to extract feats.        
        right_boxes_refine = self.right_bbox_refiner(right_roi_feats)  # Note: shape [B*100, 4, 7, 7]
        right_scores_refine = self.right_bbox_refiner_scorer(right_roi_feats)  # Note: shape [B*100, 1, 7, 7]
        logits, ker_h, ker_w = right_boxes_refine.shape[-3:]
        if self.logger is not None:
            roi_feat_sample = right_roi_feats[0, 0, :, :].detach().cpu()
            roi_feat_sample = roi_feat_sample - roi_feat_sample.min()
            roi_feat_sample /= roi_feat_sample.max()
            self.logger.add_image("roi_feat_sample", roi_feat_sample)

        batch_size = bboxes_pred.shape[0]
        x_tl_r = _bboxes_pred[..., 0].view(batch_size, -1).unsqueeze(-1)
        x_br_r = _bboxes_pred[..., 2].view(batch_size, -1).unsqueeze(-1)
        sbboxes_pred = torch.cat([bboxes_pred, x_tl_r, x_br_r], dim=-1)
        right_boxes_refine = right_boxes_refine.view(batch_size, -1, logits, ker_h, ker_w)
        refined_right_bboxes = self._right_bbox_decode(sbboxes_pred, right_boxes_refine)

        return sbboxes_pred, refined_right_bboxes, right_scores_refine.view(batch_size, -1, 1, ker_h * ker_w).permute(0, 1, 3, 2)

    def forward_keypt_det(
        self,
        left_feat: List[Tensor],
        bbox_pred: Tensor,
        keypt_predictor: nn.Sequential,
        keypt_id: str
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
        roi_feats = self.keypts_bbox_roi_extractor(left_feat, bnum_rois)  # Note: output shape is (b*100, 128, 14, 14)
        keypts_feat = keypt_predictor(roi_feats)  # Note: (b*100, num_of_class, 28, 28)
        if self.logger is not None:
            self.keypts_roi_visz[keypt_id] = roi_feats.detach()
            self.keypts_featmap_visz[keypt_id] = keypts_feat.detach()

        # TODO: put this part to prediction?
        # map_size = keypts_feat.shape[-1]
        # keypts_prob_x = ComputeSoftArgMax1d(keypts_feat.sum(dim=-2))  # Note: shape is (b*100, num_classes, 1)
        # keypts_prob_x = keypts_prob_x.unsqueeze(-1)
        # keypts_prob_y = ComputeSoftArgMax1d(keypts_feat.sum(dim=-1))
        # keypts_prob_y = keypts_prob_y.unsqueeze(-1)
        # keypts_pred = torch.cat(
        #     (keypts_prob_x, keypts_prob_y), dim=-1
        # ) / map_size  # Note: shape is (b*100, num_classes, 2)
        # batch_size, num_bboxes = bbox_pred.shape[:2]
        # num_classes, num_pts_dimension = keypts_pred.shape[-2:]
        # return keypts_pred.view(batch_size, num_bboxes, num_classes, num_pts_dimension)

        batch_size, num_bboxes = bbox_pred.shape[:2]
        num_classes = self._config['num_classes']
        featmap_size = keypts_feat.shape[-1]
        return keypts_feat.view(batch_size, num_bboxes, num_classes, featmap_size, featmap_size).sigmoid()
    
    @torch.no_grad()
    def FormatPredictionResult(
        self,
        left_bboxes_pred: Tensor,
        left_cls_scores: Tensor,
        left_objectnesses: Tensor,
        right_bboxes_pred: Tensor,
        right_scores: Tensor,
        keypt1_pred: Tensor,
        keypt2_pred: Tensor,
        batch_img_metas: Dict
    ):
        """
        Args:
            left_bboxes_pred: (b, topk, 4) shape, (tl_x, tl_y, br_x, br_y) format bbox, all in global scale.
            left_cls_scores: (b, topk, 1) shape,
            left_objectnesses: (b, topk) shape,
            right_bboxes_pred: (b, topk, ker_h * ker_w, 4). (tl_x_r, tl_y_r, br_x_r, br_y_r) format, refined right bboxes. all in global scale.
            right_scores: (b, topk, ker_h * ker_w, 1)
            keypt1_pred: (b, topk, num_classes, ker_h_keypt, ker_w_keypt)
            keypt2_pred: (b, topk, num_classes, ker_h_keypt, ker_w_keypt),
            batch_img_metas: include 'h', 'w', 'h_cam', 'w_cam' keys.

        Returns:
            detections: List of Tensor, each (N, 17) shape.
        """
        ### detection layout:
        # [clsId, left_bbox, right_bbox, keypt1, keypt2, object_confidence, stereo_confidence, keypt1_confi, keypt2_confi].
        ### number of columns of each key above:
        # 1,
        # (xn, yn, wn, hn)[n means normalized],
        # (xn_r, yn_r, wn_r, hn_r),
        # (dxn,dyn)[normalized wrt top left corner and by bbox size.],
        # (dx,dy),
        # 1
        # 1
        # 1
        # 1
        num_imgs, topk = left_bboxes_pred.shape[:2]
        ker_h_keypt, ker_w_keypt = keypt1_pred.shape[-2:]

        max_cls_scores = torch.max(left_cls_scores.detach(), dim=-1)
        clsIds = max_cls_scores[1]  # (b, topk)
        object_confidences = max_cls_scores[0] * left_objectnesses.detach()  # (b, topk)
        max_right_scores = torch.max(right_scores.detach().sigmoid().squeeze(dim=-1), dim=-1)  # similarly applied sigmoid in loss_by_stereo
        stereo_confidences = max_right_scores[0]  # (b, topk)
        # advanced indexing
        device = right_bboxes_pred.device
        refined_right_bboxes_pred = right_bboxes_pred.detach()[torch.arange(num_imgs, device=device), torch.arange(topk, device=device), max_right_scores[1]]
        keypts_pred = []
        keypts_scores = []
        for keypt_pred in [keypt1_pred, keypt2_pred]:
            device = keypt_pred.device
            keypt_pred_classified = keypt_pred.detach()[torch.arange(num_imgs, device=device), torch.arange(topk, device=device), clsIds]            
            max_keypt = torch.max(keypt_pred_classified.view(num_imgs, topk, -1), dim=-1)            
            keypt_pred_classified_bestx = (max_keypt[1] % ker_w_keypt) / ker_w_keypt
            keypt_pred_classified_besty = (max_keypt[1] // ker_w_keypt) / ker_h_keypt
            keypt_pred_classified = torch.stack([keypt_pred_classified_bestx, keypt_pred_classified_besty], dim=-1)
            keypts_pred.append(keypt_pred_classified)
            keypts_scores.append(
                torch.max(keypt_pred.view(num_imgs, topk, -1), dim=-1)[0]
            )

        detectionsBatch = []
        for indexImg in range(num_imgs):            
            maskGood = torch.logical_and(object_confidences[indexImg] > self._config['lconfidence_threshold'], stereo_confidences[indexImg] > self._config['rscore_threshold'])
            clsIds_selected = clsIds[indexImg][maskGood]
            left_bboxes_selected = left_bboxes_pred[indexImg][maskGood]
            right_bboxes_selected = refined_right_bboxes_pred[indexImg][maskGood]
            # nms final round
            if left_bboxes_selected.shape[0] > 0:
                keep_idxs = batched_nms(
                    left_bboxes_selected,
                    object_confidences[indexImg][maskGood],
                    torch.zeros_like(object_confidences[indexImg][maskGood]),
                    {'type': 'nms', 'iou_threshold': self._config['final_iou_threshold']}
                )[1]
                r_keep_idxs = batched_nms(
                    right_bboxes_selected[keep_idxs],
                    object_confidences[indexImg][maskGood][keep_idxs],
                    torch.zeros_like(object_confidences[indexImg][maskGood][keep_idxs]),
                    {'type': 'nms', 'iou_threshold': self._config['final_iou_threshold']}
                )[1]
            else:
                continue
            
            left_bboxes_selected = left_bboxes_selected[keep_idxs][r_keep_idxs]
            right_bboxes_selected = right_bboxes_selected[keep_idxs][r_keep_idxs]

            left_bboxes_selected = self.encode_bboxes(left_bboxes_selected, batch_img_metas['w_cam'], batch_img_metas['h_cam'])
            right_bboxes_selected = self.encode_bboxes(right_bboxes_selected, batch_img_metas['w_cam'], batch_img_metas['h_cam'])            
            keypt1_selected = keypts_pred[0][indexImg][maskGood][keep_idxs][r_keep_idxs]
            keypt2_selected = keypts_pred[1][indexImg][maskGood][keep_idxs][r_keep_idxs]

            object_confidence_selected = object_confidences[indexImg][maskGood][keep_idxs][r_keep_idxs]
            stereo_confidence_selected = stereo_confidences[indexImg][maskGood][keep_idxs][r_keep_idxs]
            keypt1_score = keypts_scores[0][indexImg][maskGood][keep_idxs][r_keep_idxs]
            keypt2_score = keypts_scores[1][indexImg][maskGood][keep_idxs][r_keep_idxs]
            detection = torch.cat([
                clsIds_selected.unsqueeze(-1)[keep_idxs][r_keep_idxs],
                left_bboxes_selected,
                right_bboxes_selected,
                keypt1_selected,
                keypt2_selected,
                object_confidence_selected.unsqueeze(-1),
                stereo_confidence_selected.unsqueeze(-1),
                keypt1_score.unsqueeze(-1),
                keypt2_score.unsqueeze(-1)
            ], dim=-1)
            detectionsBatch.append(detection)

        return detectionsBatch

    @staticmethod
    def encode_bboxes(bboxes: Tensor, imgWidth: int, imgHeight: int):
        """
        Args:
            bboxes: (N, 4) shape
        """
        x = (bboxes[:, 2] + bboxes[:, 0]) / 2
        y = (bboxes[:, 3] + bboxes[:, 1]) / 2
        w = (bboxes[:, 2] - bboxes[:, 0])
        h = (bboxes[:, 3] - bboxes[:, 1])
        bboxes = torch.stack([x, y, w, h], dim=1)
        bboxes[:, [0, 2]] /= imgWidth
        bboxes[:, [1, 3]] /= imgHeight
        return bboxes

    def loss_by_stereobboxdet(
        self,
        priors: Tensor,
        cls_scores: Tensor,
        bboxes: Tensor,
        right_bboxes: Tensor,
        right_scores: Tensor,
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
            bboxes: shape [B, 100, 6]. (tl_x, tl_y, br_x, br_y, tl_x_r, br_x_r) format bbox, all in global scale.
            right_bboxes: shape [B, 100, ker_h*ker_w, 4]. (tl_x_r, tl_y_r, br_x_r, br_y_r) format, refined right bboxes. all in global scale.
            right_scores: shape [B, 100, ker_h*ker_w, 1]. Scores for above right_bboxes.
            objectness: shape [B, 100].
            keypt1_pred: [B, 100, num_class, ker_h, ker_w]
            keypt2_pred: [B, 100, num_class, ker_h, ker_w]
            batch_gt_labels: include 'bboxes', 'labels', 'keypt1_masks', 'keypt2_masks' keys.
            batch_img_metas: include 'h' and 'w' keys.

        Returns:
            loss_dict: ...
            pos_masks: positive mask for the batch data.
        """        
        num_imgs = priors.shape[0]

        pos_masks, cls_targets, obj_targets, bbox_targets, indices_bbox_targets, num_pos_per_img = [], [], [], [], [], []
        for indexInBatch in range(len(batch_gt_labels)):
            (
                pos_mask,
                cls_target,
                obj_target,  # If it is a thing, no matter what class, objectness_target is 1.0
                bbox_target,
                indices_bbox_target,
                num_pos_one  # number of positive gt target in each image.
            ) = self._get_targets_stereo_single(
                priors.detach()[indexInBatch],
                cls_scores.detach()[indexInBatch],
                bboxes.detach()[indexInBatch],
                objectness.detach()[indexInBatch],
                batch_gt_labels[indexInBatch],
                img_metas=batch_img_metas
            )
            pos_masks.append(pos_mask)
            cls_targets.append(cls_target)
            obj_targets.append(obj_target)
            bbox_targets.append(bbox_target)
            indices_bbox_targets.append(indices_bbox_target)
            num_pos_per_img.append(num_pos_one)

        num_pos = torch.tensor(
            sum(num_pos_per_img),
            dtype=torch.float,
            device=cls_scores.device
        )

        pos_masks = torch.cat(pos_masks, 0)
        cls_targets = torch.cat(cls_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        bbox_targets = torch.cat(bbox_targets, 0)
        indices_bbox_targets = torch.cat(indices_bbox_targets, 0)        

        if num_pos > 0:
            rbboxes_targets = torch.cat([
                bbox_targets[:, 4].unsqueeze(-1),
                bbox_targets[:, 1].unsqueeze(-1),
                bbox_targets[:, 5].unsqueeze(-1),
                bbox_targets[:, 3].unsqueeze(-1)
            ], dim=1)
            
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
            # keypoints
            loss_keypts = []
            classSelection = torch.argmax(cls_scores, dim=2)
            featmap_size = keypt1_pred.shape[-1]
            mask_target_visz = {}
            mask_preds_visz = {}
            for keypt_pred, lossfunc_keypt, indexKeypt in zip([keypt1_pred, keypt2_pred], [self.loss_keypt1, self.loss_keypt2], [1, 2]):
                num_batch, num_priors = keypt_pred.shape[:2]
                pos_bboxes_preds_list, pos_assigned_gt_indices_list, keypt_masks_list = [], [], []
                for indexImg in range(num_batch):
                    # collect targets in each img.
                    indices_bbox_targets_oneimg = indices_bbox_targets[sum(num_pos_per_img[:indexImg]):sum(num_pos_per_img[:indexImg + 1]), :].to(torch.int).squeeze(-1)
                    pos_assigned_gt_indices_list.append(indices_bbox_targets_oneimg)
                    pos_bboxes_pred_oneimg = selected_sbboxes[sum(num_pos_per_img[:indexImg]):sum(num_pos_per_img[:indexImg + 1]), :4].detach()
                    pos_bboxes_preds_list.append(pos_bboxes_pred_oneimg)
                    # select masks for keypoint
                    keypt_key = "keypt2_masks" if indexKeypt == 2 else "keypt1_masks"
                    keypt_masks_oneimg = batch_gt_labels[indexImg][keypt_key]
                    keypt_masks_list.append(BitmapMasks(keypt_masks_oneimg.cpu().numpy(), height=batch_img_metas['h'], width=batch_img_metas['w']))                          
                mask_targets = mask_target(
                    pos_bboxes_preds_list,
                    pos_assigned_gt_indices_list,
                    keypt_masks_list,
                    Config({"mask_size": featmap_size, 'soft_mask_target': True})
                )
                pos_class_labels = classSelection.view(-1)[pos_masks]
                keypt_mask_preds = keypt_pred.view(-1, self._config["num_classes"], featmap_size, featmap_size)[pos_masks]
                keypt_mask_preds = keypt_mask_preds[torch.arange(0, keypt_mask_preds.shape[0]), pos_class_labels.to(torch.int)]

                loss_keypts.append(lossfunc_keypt(keypt_mask_preds, mask_targets))
                if self.logger is not None:
                    mask_target_visz[str(indexKeypt)] = mask_targets
                    mask_preds_visz[str(indexKeypt)] = keypt_mask_preds

            if self.logger is not None:
                for key in ['1', '2']:
                    keypt_feat = mask_preds_visz[key][0].detach().cpu()
                    keypt_feat = keypt_feat - keypt_feat.min()
                    keypt_feat *= 255 / keypt_feat.max()
                    keypt_feat = F.interpolate(keypt_feat.unsqueeze(0).unsqueeze(0), size=(720, 720), mode='bilinear', align_corners=False)
                    self.logger.add_image("keypt_feat_{}_sample".format(key), keypt_feat.to(torch.uint8).squeeze())
                    keypt_target = mask_target_visz[key][0].detach().cpu()
                    keypt_target = keypt_target - keypt_target.min()
                    keypt_target *= 255 / keypt_target.max()
                    keypt_target = F.interpolate(keypt_target.unsqueeze(0).unsqueeze(0), size=(720, 720), mode='bilinear', align_corners=False)
                    self.logger.add_image("keypt_target_{}_sample".format(key), keypt_target.to(torch.uint8).squeeze())
                    keypt_roi = self.keypts_roi_visz[key][pos_masks][0].mean(dim=0).detach().cpu()
                    keypt_roi = keypt_roi - keypt_roi.min()
                    keypt_roi *= 255 / keypt_roi.max()
                    keypt_roi = F.interpolate(keypt_roi.unsqueeze(0).unsqueeze(0), size=(720, 720), mode='bilinear', align_corners=False)
                    self.logger.add_image("keypt_roi_{}_sample".format(key), keypt_roi.to(torch.uint8).squeeze())
        else:
            # if no gt, then not update
            # see https://github.com/open-mmlab/mmdetection/issues/7298
            loss_keypts = [
                keypt1_pred.sum() * 0,
                keypt2_pred.sum() * 0
            ]
            loss_rbbox = right_bboxes.sum() * 0
            loss_rscore = right_scores.sum() * 0
        
        loss_dict = {
            'loss_rbbox': loss_rbbox,
            'loss_rscore': loss_rscore,
            'loss_keypt1': loss_keypts[0],
            'loss_keypt2': loss_keypts[1]
        }
        torch.cuda.synchronize()

        return  loss_dict, pos_masks.reshape(num_imgs, -1), bboxes

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
            gt_labels (Dict): It includes 'bboxes' (num_instances, 11) and 'labels' (num_instances,) and 'keypt1_masks' and 'keypt2_masks'.
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
            indices_bbox_target = cls_scores.new_zeros((0, 1))
            return (pos_mask, cls_target, obj_target, bbox_target, indices_bbox_target, 0)

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
        indices_bbox_target = sampling_result.pos_bboxes_indices
        pos_mask = torch.zeros_like(objectness).to(torch.bool)
        pos_mask[pos_inds] = True
        return (pos_mask, cls_target, obj_target, bbox_target, indices_bbox_target, num_pos_per_img)

    def loss_by_bboxdet(
        self,
        cls_scores: Tuple[Tensor],
        bbox_preds: Tuple[Tensor],
        objectnesses: Tuple[Tensor],
        gt_labels: List[Dict],
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

        pos_masks, cls_targets, obj_targets, bbox_targets, num_pos_imgs = [], [], [], [], []
        for indexInBatch in range(len(gt_labels)):
            (
                pos_mask,
                cls_target,
                obj_target,
                bbox_target,
                num_pos_img
            ) = self._get_targets_single(
                flatten_priors.unsqueeze(0).repeat(num_imgs, 1, 1)[indexInBatch],
                flatten_cls_preds.detach()[indexInBatch],
                flatten_bboxes.detach()[indexInBatch],
                flatten_objectness.detach()[indexInBatch],
                gt_labels[indexInBatch],
                img_meta=batch_img_metas)
            pos_masks.append(pos_mask)
            cls_targets.append(cls_target)
            obj_targets.append(obj_target)
            bbox_targets.append(bbox_target)
            num_pos_imgs.append(num_pos_img)

        num_pos = torch.tensor(
            sum(num_pos_imgs),
            dtype=torch.float,
            device=flatten_cls_preds.device)
        num_total_samples = max(num_pos, 1.0)
        
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
        bboxes_targets_selected = batch_gt_bboxes.view(-1, 1, 4).expand(-1, candidates_k, -1)
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


