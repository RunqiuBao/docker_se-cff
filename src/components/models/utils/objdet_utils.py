import torch.nn as nn
import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Sequence, Tuple, List, Dict, Union, Optional
import time
import torch.profiler

from mmdet.structures.bbox import cat_boxes
from mmdet.models.task_modules.prior_generators import MlvlPointGenerator
from mmdet.models.task_modules.samplers import PseudoSampler
from mmdet.models.task_modules.assigners import SimOTAAssigner
from mmengine.structures import InstanceData
from mmdet.registry import TASK_UTILS
from mmcv.ops import batched_nms


@torch.no_grad()
def SelectTopkCandidates_single(
    cls_scores: Tuple[Tensor],
    bbox_preds: Tuple[Tensor],
    objectnesses: Tuple[Tensor],
    img_metas: Dict,
    config: Dict,
    prior_generator: MlvlPointGenerator
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
    mlvl_priors = prior_generator.grid_priors(
        featmap_sizes,
        dtype=cls_scores[0].dtype,
        device=cls_scores[0].device,
        with_stride=True
    )

    nms_pre = config.get('nms_pre', -1)
    mlvl_bbox_preds = []
    mlvl_valid_priors = []
    mlvl_confidences = []
    mlvl_cls_scores = []
    mlvl_objectness = []
    level_ids = []
    for level_idx, (cls_score, bbox_pred, objectness, priors) in enumerate(zip(cls_scores, bbox_preds, objectnesses, mlvl_priors)):
        bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
        cls_score = cls_score.permute(1, 2, 0).reshape(-1, config['num_classes']).sigmoid()
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
    bboxes = _bbox_decode(priors, bbox_pred)

    results = InstanceData()
    results.bboxes = bboxes
    results.scores = torch.cat(mlvl_confidences)
    results.cls_scores = torch.cat(mlvl_cls_scores)
    results.priors = priors
    results.objectness = torch.cat(mlvl_objectness)
    results.level_ids = torch.cat(level_ids)
    # filter small size bboxes
    if config.get('min_bbox_size', -1) >= 0:
        w = results.bboxes[:, 2] - results.bboxes[:, 0]
        h = results.bboxes[:, 3] - results.bboxes[:, 1]
        valid_mask = (w > config['min_bbox_size']) & (h > config['min_bbox_size'])
        if not valid_mask.all():
            results = results[valid_mask]
    
    det_bboxes, keep_idxs = batched_nms(
        results.bboxes,
        results.scores,
        results.level_ids,
        {'type': 'nms', 'iou_threshold': config['nms_iou_threshold']}
    )
    results = results[keep_idxs]
    if results.bboxes.shape[0] < config['num_topk_candidates']:
        # patching missing length with zeros
        len_missing = config['num_topk_candidates'] - results.bboxes.shape[0]
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
        results = results[:config['num_topk_candidates']]
    return results.cls_scores, results.bboxes, results.objectness, results.priors


def _bbox_decode(priors: Tensor, bbox_preds: Tensor) -> Tensor:
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
def SelectTopkCandidates(
    cls_scores: Tuple[Tensor],
    bbox_preds: Tuple[Tensor],
    objectnesses: Tuple[Tensor],
    img_metas: Dict,
    config: Dict
):
    """
    select topk candidates for a batch data.
    """
    batch_size = cls_scores[0].shape[0]
    bboxes_selected, cls_scores_selected, objectness_selected, priors_selected = [], [], [], []
    prior_generator = MlvlPointGenerator(config["strides"], offset=0)
    for indexD in range(batch_size):
        cls_scores_one = [cls_score[indexD] for cls_score in cls_scores]
        bbox_preds_one = [bbox_pred[indexD] for bbox_pred in bbox_preds]
        objectness_one = [objectness[indexD] for objectness in objectnesses]
        cls_scores_one, bboxes_one, objectness_one, priors_one = SelectTopkCandidates_single(
            cls_scores_one,
            bbox_preds_one,
            objectness_one,
            img_metas=img_metas,
            config=config,
            prior_generator=prior_generator
        )
        if config['freeze_leftobjdet']:
            # Note: clamp tensor will only make the clampped pixels not differential.
            # Note: prevent stereo detection stuck
            bboxes_one[:, [0, 2]] = bboxes_one[:, [0, 2]].clamp(min=0, max=img_metas["w"])
            bboxes_one[:, [1, 3]] = bboxes_one[:, [1, 3]].clamp(min=0, max=img_metas["h"])            
        bboxes_selected.append(bboxes_one.unsqueeze(0))
        cls_scores_selected.append(cls_scores_one.unsqueeze(0))
        objectness_selected.append(objectness_one.squeeze(-1).unsqueeze(0))
        priors_selected.append(priors_one.unsqueeze(0))
    return torch.cat(cls_scores_selected, dim=0), torch.cat(bboxes_selected, dim=0), torch.cat(objectness_selected, dim=0), torch.cat(priors_selected, dim=0)


@torch.no_grad()
def _get_targets_single(
    priors: Tensor,
    cls_scores: Tensor,
    bboxes: Tensor,
    objectness: Tensor,
    gt_labels: Dict,
    config: Dict,
    assigner: SimOTAAssigner,
    sampler: PseudoSampler
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
        assigner: SimOTAAssigner in case of YoloX
        sampler: PseudoSampler in case of YoloX
    """
    num_priors = priors.shape[0]
    num_gts = gt_labels['bboxes'].shape[0]

    # no target
    if num_gts == 0:
        cls_target = cls_scores.new_zeros((0, config['num_classes']))
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

    starttime = time.time()
    # use SimOTA dynamic assigner, same as yolox
    assign_result = assigner.assign(
        pred_instances=pred_instances,
        gt_instances=gt_instances
    )
    # with torch.profiler.profile(
    #     activities=[
    #         torch.profiler.ProfilerActivity.CPU,
    #         torch.profiler.ProfilerActivity.CUDA],
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
    # ) as prof:
    #     assign_result = self.assigner.assign(
    #         pred_instances=pred_instances,
    #         gt_instances=gt_instances
    #     )
    # if not self.training:
    #     print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    # use a pesudo sampler to get all results. just for using mmdet util's api.
    starttime = time.time()
    sampling_result = sampler.sample(
        assign_result,
        pred_instances,
        gt_instances
    )
    # print("----- pesudo sampler: {}".format(time.time() - starttime))

    pos_inds = sampling_result.pos_inds
    num_pos_per_img = pos_inds.size(0)

    pos_ious = assign_result.max_overlaps[pos_inds]
    # Yolox: IoU aware classification scores
    cls_target = F.one_hot(sampling_result.pos_gt_labels, config['num_classes']) * pos_ious.unsqueeze(-1)
    obj_target = torch.zeros_like(objectness).unsqueeze(-1)
    obj_target[pos_inds] = 1
    bbox_target = sampling_result.pos_gt_sbboxes
    indices_bbox_target = sampling_result.pos_bboxes_indices
    pos_mask = torch.zeros_like(objectness).to(torch.bool)
    pos_mask[pos_inds] = True
    
    return (pos_mask, cls_target, obj_target, bbox_target, indices_bbox_target, num_pos_per_img)


@torch.no_grad
def SelectTargets(
    priors: Tensor,
    cls_scores: Tensor,
    bboxes: Tensor,
    objectness: Tensor,
    batch_gt_labels: Dict,
    config: Dict
):
    """
    For yoloX.

    Args:
        priors: shape [B, 100, 4],
        cls_scores: shape [B, 100, num_class]
        bboxes: shape [B, 100, 6]. (tl_x, tl_y, br_x, br_y, tl_x_r, br_x_r) format bbox, all in global scale.
        objectness: shape [B, 100].
        batch_gt_labels: include 'bboxes', 'labels', 'keypt1_masks', 'keypt2_masks' (or 'leftmasks') keys.
        batch_img_metas: include 'h' and 'w' keys.
    
    Returns:

    """
    pos_masks, cls_targets, obj_targets, bbox_targets, indices_bbox_targets, batch_num_pos_per_img = [], [], [], [], [], []
    assigner = TASK_UTILS.build({'type': 'SimOTAAssigner', 'center_radius': 2.5})
    sampler = PseudoSampler()
    for indexInBatch in range(len(batch_gt_labels)):
        (
            pos_mask,
            cls_target,
            obj_target,  # If it is a thing, no matter what class, objectness_target is 1.0
            bbox_target,
            indices_bbox_target,
            num_pos_one  # number of positive gt target in each image.
        ) = _get_targets_single(
            priors.detach()[indexInBatch],
            cls_scores.detach()[indexInBatch],
            bboxes.detach()[indexInBatch],
            objectness.detach()[indexInBatch],
            batch_gt_labels[indexInBatch],
            config=config,
            assigner=assigner,
            sampler=sampler
        )
        pos_masks.append(pos_mask)
        cls_targets.append(cls_target)
        obj_targets.append(obj_target)
        bbox_targets.append(bbox_target)
        indices_bbox_targets.append(indices_bbox_target)
        batch_num_pos_per_img.append(num_pos_one)
    # print("----- time sub sub loss stereo: {}".format(time.time() - starttime))

    num_pos = torch.tensor(
        sum(batch_num_pos_per_img),
        dtype=torch.float,
        device=cls_scores.device
    )

    pos_masks = torch.cat(pos_masks, 0)
    cls_targets = torch.cat(cls_targets, 0)
    obj_targets = torch.cat(obj_targets, 0)
    bbox_targets = torch.cat(bbox_targets, 0)
    indices_bbox_targets = torch.cat(indices_bbox_targets, 0)
    return num_pos, pos_masks, cls_targets, bbox_targets, indices_bbox_targets, batch_num_pos_per_img
