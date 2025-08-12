import numpy
import torch
from typing import Optional, List, Dict, Tuple
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
from thop import profile

from mmdet.registry import MODELS, TASK_UTILS
from mmengine.model import initialize
from mmcv.cnn import ConvModule

from . import losses
from .event_stereo_object_detection import DecodeKeypts

import logging
logger = logging.getLogger(__name__)


class LocalTrackingHead(nn.Module):
    def __init__(
        self,
        network_cfg: dict,
        loss_cfg: dict,
        is_freeze: bool,
        **kwargs
    ):
        super(LocalTrackingHead, self).__init__()
        self._config = network_cfg
        self._config["is_freeze"] = is_freeze

        self._config["loss_cfg"] = loss_cfg

        # objdet tools
        self.backbone = MODELS.build(
            {
                'type': 'ResNeXt',
                'depth': 101,
                'num_stages': 4,
                'out_indices': (0, 1, 2, 3),
                'frozen_stages': 1,
                'norm_cfg': {'type': 'BN', 'requires_grad': True},
                'norm_eval': True,
                'style': 'pytorch',
                'groups': 64,
                'base_width': 4,
                'in_channels': self._config['in_channels'],
            }
        )
        self.neck = MODELS.build(
            {
                'type': 'FPN',
                'in_channels': [256, 512, 1024, 2048],
                'out_channels': self._config['channels_roi_extraction'],
                'num_outs': 5
            }
        )

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
            self.shared_convs,
            self.shared_fcs,
            last_layer_dim
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

    @staticmethod
    def ComputeCostProfile(model):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        h, w = 480, 672
        input_feats = torch.randn(4, 10, int(h / 8), int(w / 8)).to(device)
        bboxes = [torch.randn(120, 4).to(device)]
        flow_prior = torch.randn(4, 2, h, w).to(device)
        batch_img_metas = {"h": h, "w": w}
        model = model.to(device)
        flops, numParams = profile(model, inputs=(input_feats,  bboxes, flow_prior, batch_img_metas), verbose=False)
        return flops, numParams

    def predict(
        self,
        event_voxel: Tensor,
        ref_bboxes: List[Tensor],
        flow_prior: Tensor,
        batch_img_metas: Dict,
        bbox_move_anchor_ticks: List[float],
        bbox_expand_factor: float
    ) -> Tuple[List[Optional[Tensor]], List[Optional[Tensor]], List[Optional[Tensor]], List[Optional[Tensor]]]:
        """
        Args:
            event_voxel: shape is [B, 10, h, w]
            ref_bboxes: list of B tensors of shape [?, 4]. [tl_x, tl_y, br_x, br_y] format bbox, all in global scale.
            flow_prior: [B, 2, h, w] shape.
            bbox_move_anchor_ticks: from 0.5 to 2.0, use multiple ticks to resize the bboxes horizontally and vertically to search for different regions.

        Returns:
            bboxes_priors: shape [B, N, 6]. format [tl_x, tl_y, br_x, br_y, tl_x_r, br_x_r] rough stereo bbox
            refined_bboxes: shape [B, N, ker_h * ker_w, 4]. Corresponding refined right bboxes.
            refined_scores: shape [B, N, ker_h * ker_w, 1]. Corresponding scores.
            predicted_keypts:
        """
        list_bboxes_priors, list_refined_bboxes, list_refined_scores, list_predicted_keypts = [], [], [], []
        
        input_feats = self.backbone(event_voxel)
        input_feats = self.neck(input_feats)
        for indexInBatch, bboxes_pred in enumerate(ref_bboxes):
            # starttime = time.time()
            if bboxes_pred.shape[0] == 0:
                # No detections in left
                list_refined_bboxes.append(None)
                list_refined_scores.append(None)
                list_predicted_keypts.append(None)
                continue
            bboxes_pred = torch.unsqueeze(bboxes_pred, dim=0)

            # enlarge bboxes
            _bboxes_pred = bboxes_pred.clone()  # initial warped bboxes for right side target.
            variation_size = len(bbox_move_anchor_ticks)
            variation_srclist = torch.tensor(bbox_move_anchor_ticks, dtype=torch.float32, device=_bboxes_pred.device)
            variations_x = variation_srclist.repeat(variation_size)
            variations_y = variation_srclist.repeat_interleave(variation_size)
            
            varitions_lxrx = torch.stack([-(torch.ones_like(variations_x) * 2.0 - variations_x), variations_x], dim=1)
            hdwboxes = bbox_expand_factor * (_bboxes_pred[..., 2] - _bboxes_pred[..., 0]) / 2.0
            cwbboxes = (_bboxes_pred[..., 2] + _bboxes_pred[..., 0]) / 2.0
            hdwboxes = hdwboxes.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, variation_size**2, 2)  # (1, N, k*k, 2)
            cwbboxes = cwbboxes.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, variation_size**2, 2)  # (1, N, k*k, 2)
            wbboxes = cwbboxes + varitions_lxrx * hdwboxes

            variations_tyby = torch.stack([-(torch.ones_like(variations_y) * 2.0 - variations_y), variations_y],  dim=1)
            dhwboxes = bbox_expand_factor * (_bboxes_pred[..., 3] - _bboxes_pred[..., 1]) / 2.0
            chbboxes = (_bboxes_pred[..., 3] + _bboxes_pred[..., 1]) / 2.0
            dhwboxes = dhwboxes.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, variation_size**2, 2)  # (1, N, k*k, 2)
            chbboxes = chbboxes.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, variation_size**2, 2)
            hbboxes = chbboxes + variations_tyby * dhwboxes

            _bboxes_pred = _bboxes_pred.unsqueeze(2).repeat(1, 1, variation_size**2, 1)  # [1, N, variation_size, 4]
            bboxes_priors = _bboxes_pred.clone().view(1, -1, 4)  # [1, N * variation_size, 4]

            _bboxes_pred[:, :, :, [0, 2]] = wbboxes[:, :, :, [0, 1]]
            _bboxes_pred[:, :, :, [1, 3]] = hbboxes[:, :, :, [0, 1]]
            _bboxes_pred = _bboxes_pred.view(1, -1, 4)
            
            num_detections = bboxes_pred.shape[1]
            batch_number = torch.arange(_bboxes_pred.shape[0]).unsqueeze(1).repeat(1, num_detections * variation_size**2).flatten().unsqueeze(-1).to(_bboxes_pred.device)

            # extract right bbox roi feature
            xindi = ((bboxes_pred[..., 0] + bboxes_pred[..., 2]) / 2).to(torch.int).clamp(0, batch_img_metas['w'] - 1).squeeze()
            yindi = ((bboxes_pred[..., 1] + bboxes_pred[..., 3]) / 2).to(torch.int).clamp(0, batch_img_metas['h'] - 1).squeeze()

            bbox_flows = flow_prior[indexInBatch][:, yindi, xindi].unsqueeze(0).unsqueeze(-1).repeat(1, 1, variation_size**2).view(1, 2, -1)

            _bboxes_pred[..., [0, 2]] += bbox_flows[:, 0, :].unsqueeze(-1).expand(1, -1, 2)
            _bboxes_pred[..., [1, 3]] += bbox_flows[:, 1, :].unsqueeze(-1).expand(1, -1, 2)
            rois_targets = _bboxes_pred.reshape(-1, 4)
            rois_targets = torch.cat((batch_number, rois_targets), dim=1)

            roi_feats = self.bbox_roi_extractor(input_feats, rois_targets)  # Note: Based on the bbox size to decide from which level to extract feats.
            # print("stereoNet time cost (until roi extract): {}".format(time.time() - starttime))
            # starttime = time.time()

            feats_hidden = roi_feats.flatten(1)
            for fc in self.shared_fcs:
                feats_hidden = self.relu(fc(feats_hidden))
            cls_score = self.fc_cls(feats_hidden)
            refined_bboxes = self.fc_reg(feats_hidden)
            predicted_keypts = self.fc_keypts(feats_hidden)

            # starttime = time.time()
            # if self.logger is not None:
            #     for indexInstance in range(right_roi_feats.shape[0]):
            #         roi_feat_sample = torch.mean(right_roi_feats[indexInstance, :, :, :], dim=0).detach().cpu()
            #         roi_feat_sample = roi_feat_sample - roi_feat_sample.min()
            #         roi_feat_sample /= roi_feat_sample.max()
            #         self.logger.add_image("roi_feat_sample{}".format(indexInstance), roi_feat_sample)

            bboxes_priors = bboxes_priors.view(1, num_detections, variation_size**2, 4)
            list_bboxes_priors.append(bboxes_priors)  # (1, num_detections, k*k, 6)
            list_refined_bboxes.append(refined_bboxes)
            list_refined_scores.append(cls_score)
            list_predicted_keypts.append(predicted_keypts)

        return list_bboxes_priors, list_refined_bboxes, list_refined_scores, list_predicted_keypts

    @staticmethod
    def kpts_decode(anchor_points, pred_kpts):
        """Decodes predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y
    
    @torch.no_grad()
    def extract_inference_results(
        self,
        bboxes_priors: Tensor,
        refined_bboxes: Tensor,
        refined_scores: Tensor,
        predicted_keypts: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Extract inference results from the model's predictions.
        Args:
            bboxes_priors: (num_detections, variation_size_squared, 4) shape, containing the right priors from preivous network.
            refined_bboxes: (num_detections*variation_size_squared, num_classes, 4) shape.
            refined_scores: (num_detections*variation_size_squared, num_classes+1) shape.
            predicted_keypts: (num_detections*variation_size_squared, num_classes, num_keypts*3) shape.
        
        Returns:
            mask_nonbackground: (num_detections,) shape.
            refined_bboxes: (<num_detections>, 4) shape, <num_detections> == mask_nonbackground.sum().
            refined_scores: (<num_detections>, 1) shape.
            predicted_keypts: (<num_detections>, num_keypts*3) shape.
        """
        num_detections, variation_size_squared = bboxes_priors.shape[:2]
        num_classes = refined_bboxes.shape[-2]
        bboxes_priors = bboxes_priors.view(num_detections, variation_size_squared, 4)

        refined_bboxes = refined_bboxes.view(num_detections, variation_size_squared, num_classes, 4)
        refined_scores = refined_scores.view(num_detections, variation_size_squared, (num_classes + 1))
        predicted_keypts = predicted_keypts.view(num_detections, variation_size_squared, num_classes, 3 * self._config["max_num_keypoints"])

        best_scores_of_classes, best_class_labels = torch.max(F.softmax(refined_scores, dim=-1), dim=-1)
        indices_highest_score = torch.argmax(best_scores_of_classes, dim=1)
        batchIndices = torch.arange(num_detections)
        class_labels_pred = best_class_labels[batchIndices, indices_highest_score]
        mask_nonbackground = class_labels_pred != num_classes
        if mask_nonbackground.sum() == 0:
            print("Warning: all detections are classified as bkground, no valid detections.")
            return mask_nonbackground, None, None, None
        else:
            bboxes_priors_nobkg = bboxes_priors[mask_nonbackground]
            refined_bboxes_nobkg = refined_bboxes[mask_nonbackground]
            predicted_keypts_nobkg = predicted_keypts[mask_nonbackground]
            class_labels_pred_nobkg = class_labels_pred[mask_nonbackground]
            # select the highest score as confidence score
            refined_scores_pred = best_scores_of_classes[torch.arange(num_detections), indices_highest_score]
            indices_highest_score_nobkg = indices_highest_score[mask_nonbackground]
            refined_scores_pred = refined_scores_pred[mask_nonbackground]
            # select data based on the class label
            refined_bboxes_nobkg = refined_bboxes_nobkg[
                torch.arange(mask_nonbackground.sum()).view(-1, 1).expand(-1, variation_size_squared),
                torch.arange(variation_size_squared).unsqueeze(0),
                class_labels_pred_nobkg.view(mask_nonbackground.sum(), 1).expand(-1, variation_size_squared)
            ]
            refined_bboxes_nobkg = refined_bboxes_nobkg[torch.arange(mask_nonbackground.sum()), indices_highest_score_nobkg]
            refined_bboxes_nobkg_decoded = self.bbox_coder.decode(bboxes_priors_nobkg[:, 0, :], refined_bboxes_nobkg.view(-1, 4))
            refined_bboxes_nobkg[:, [0, 2]] = refined_bboxes_nobkg_decoded[:, [0, 2]]
            refined_bboxes_nobkg = refined_bboxes_nobkg.view(mask_nonbackground.sum(), 4)
            # select best keypoints
            predicted_keypts_nobkg = predicted_keypts_nobkg[
                torch.arange(mask_nonbackground.sum()).view(-1, 1).expand(-1, variation_size_squared),
                torch.arange(variation_size_squared).unsqueeze(0),
                class_labels_pred_nobkg.view(mask_nonbackground.sum(), 1).expand(-1, variation_size_squared)
            ]
            predicted_keypts_nobkg = predicted_keypts_nobkg[torch.arange(mask_nonbackground.sum()), indices_highest_score_nobkg]
            predicted_keypts_nobkg = predicted_keypts_nobkg.clone()
            for indexKeypt in range(self._config["max_num_keypoints"]):
                predicted_keypts_nobkg[:, (3 * indexKeypt):(2 + 3 * indexKeypt)] = DecodeKeypts(
                    bboxes_priors_nobkg[:, 0, :],
                    predicted_keypts_nobkg[:, (3 * indexKeypt):(2 + 3 * indexKeypt)],
                )
            return mask_nonbackground, refined_bboxes_nobkg, refined_scores_pred, predicted_keypts_nobkg.view(mask_nonbackground.sum(), -1, 3)
