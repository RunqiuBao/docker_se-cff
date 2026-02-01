import torch.nn as nn
import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Sequence, Tuple, List, Dict, Union, Optional
import time
import torch.profiler
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from dataclasses import dataclass

from mmdet.structures.bbox import cat_boxes
from mmdet.models.task_modules.prior_generators import MlvlPointGenerator
from mmdet.models.task_modules.samplers import PseudoSampler
from mmdet.models.task_modules.assigners import SimOTAAssigner
from mmengine.structures import InstanceData
from mmdet.registry import TASK_UTILS
from mmcv.ops import batched_nms


def evaluate_results_with_gt(
    pred_bboxes: torch.Tensor,
    pred_labels: torch.Tensor,
    gt_bboxes: torch.Tensor,
    gt_labels: torch.Tensor,
    iou_threshold: float,
):
    """
    Args:
        pred_bboxes: (N, 8) tensor, [x1, y1, x2, y2, x1_r, y1_r, x2_r, y2_r]
        pred_labels: (N,) tensor
        gt_bboxes: (M, 8) tensor, [x1, y1, x2, y2, x1_r, y1_r, x2_r, y2_r]
        gt_labels: (M,) tensor
    """
    ious = compute_ious_pertarget(pred_bboxes[:, :4], gt_bboxes[:, :4])   # (N, M)
    ious_r = compute_ious_pertarget(pred_bboxes[:, 4:], gt_bboxes[:, 4:])   # (N, M)
    ious_min = torch.min(ious, ious_r)

    matched_gt = set()
    TP, FP, FN = 0, 0, 0
    for idxP in range(pred_bboxes.shape[0]):
        iou_per_pred = ious_min[idxP]   # (M,)
        max_iou, max_iou_idx = torch.max(iou_per_pred, dim=0)
        if max_iou >= iou_threshold and pred_labels[idxP] == gt_labels[max_iou_idx] and (max_iou_idx.item() not in matched_gt):
            TP += 1
            matched_gt.add(max_iou_idx.item())
        else:
            FP += 1
    FN = gt_bboxes.shape[0] - len(matched_gt)
    return TP, FP, FN


def compute_ious_pertarget(bboxes_pred: torch.Tensor, bboxes_ref: torch.Tensor) -> torch.Tensor:
    """
    Args:
        bboxes_pred: (N, 4) tensor [x1, y1, x2, y2]
        bboxes_ref: (K, 4) tensor [x1, y1, x2, y2]

    Returns: 
        ious: (N, K), tensor of IoUs.
    """
    # Intersection top-left & bottom-right
    lt = torch.max(bboxes_pred[:, None, :2], bboxes_ref[:, :2])   # (N, K, 2)
    rb = torch.min(bboxes_pred[:, None, 2:], bboxes_ref[:, 2:])   # (N, K, 2)

    wh = (rb - lt).clamp(min=0)   # (N, K, 2)
    inter = wh[:, :, 0] * wh[:, :, 1]   # (N, K)

    # Areas
    area1 = ((bboxes_pred[:, 2] - bboxes_pred[:, 0]) *
             (bboxes_pred[:, 3] - bboxes_pred[:, 1]))[:, None]     # (N, 1)
    area2 = ((bboxes_ref[:, 2] - bboxes_ref[:, 0]) *
             (bboxes_ref[:, 3] - bboxes_ref[:, 1]))              # (K,)

    union = area1 + area2 - inter
    ious = inter / union.clamp(min=1e-6)

    return ious


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


@torch.no_grad
def EvaluateObjDetPerformance(preds: List[dict], targets: List[dict]):
    """
    For example,
        Args:
            preds = [
                {
                    "boxes": torch.tensor([[100, 100, 200, 200], [200, 200, 300, 300]]),  # (x1, y1, x2, y2)
                    "scores": torch.tensor([0.9, 0.8]),
                    "labels": torch.tensor([1, 1])
                }
            ],
            targets = [
                {
                    "boxes": torch.tensor([[102, 98, 198, 202], [200, 200, 300, 300]]),
                    "labels": torch.tensor([1, 1])
                }
            ]

        Returns:
            {'map': tensor(0.9112),
            'map_50': tensor(1.),
            'map_75': tensor(1.),
            'map_small': tensor(-1.),
            'map_medium': tensor(-1.),
            'map_large': tensor(0.9112),
            'mar_1': tensor(0.6000),
            'mar_10': tensor(0.9333),
            'mar_100': tensor(0.9333),
            'mar_small': tensor(-1.),
            'mar_medium': tensor(-1.),
            'mar_large': tensor(0.9333),
            'map_per_class': tensor(-1.),
            'mar_100_per_class': tensor(-1.)}
    """
    metric = MeanAveragePrecision(iou_type="bbox")
    metric.update(preds, targets)
    return metric.compute()



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


def AllocateHypothesesToTargets(
    listfeat_cls_score: list[Tensor],
    listfeat_bbox_pred: list[Tensor],
    targets: List[Tensor],
    imageHeight,
    imageWidth,
    num_classes,  # number of object types
    nms_pred: int = 2000,
    max_hypotheses_per_img: int = 1000,
    min_iou_with_target: float = 0.0,
    min_bbox_size: float = 0.,
    prior_generator = None,
    bbox_coder = None,
) -> List[RPNHypothesesGroup]:
    """
    Allocate K best hypotheses to each target.
    """
    if prior_generator is None:
        anchors_ratios = [0.5, 1.0, 2.0]
        prior_generator = TASK_UTILS.build(
            {
                'type': 'AnchorGenerator',
                'scales': [8],
                'ratios': anchors_ratios,
                'strides': [4, 8, 16, 32]
            }
        )
    if bbox_coder is None:
        bbox_coder = TASK_UTILS.build(
            {
                'type': 'DeltaXYWHBBoxCoder',
                'target_means': [0.0, 0.0, 0.0, 0.0],
                'target_stds': [1.0, 1.0, 1.0, 1.0]
            }
        )

    cls_out_channels = num_classes
    num_images_in_batch = len(targets)
    num_levels = len(listfeat_cls_score)

    featmap_sizes = [listfeat_cls_score[i].shape[-2:] for i in range(num_levels)]
    mlvl_priors = prior_generator.grid_priors(
        featmap_sizes,
        dtype=listfeat_cls_score[0].dtype,
        device=listfeat_cls_score[0].device)

    reg_dim = bbox_coder.encode_size
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
            cls_score_onelvl = cls_score_onelvl.permute(1, 2, 0).reshape(-1, cls_out_channels)
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
        bboxes_xyxy = bbox_coder.decode(priors, bbox_pred, max_shape=(imageHeight, imageWidth))
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


def WarpBboxes(
    left_bboxes: List[Tensor],
    disp_prior: Tensor,
    imageHeight: int,
    imageWidth: int,
):
    """
    Args:
        left_bboxes: batch of tensor, bboxes in left images.
        disp_prior: (B, h, w).
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


def ExtractStereoInferenceResults(
    sbboxes_priors: Tensor,
    refined_right_bboxes: Tensor,
    refined_right_scores: Tensor,
    right_keypts_pred: Tensor,
    max_num_keypoints: int,
    bbox_coder=None, 
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
    if bbox_coder is None:
        bbox_coder = TASK_UTILS.build(
            {
                'type': 'DeltaXYWHBBoxCoder',
                'target_means': [0.0, 0.0, 0.0, 0.0],
                'target_stds': [1.0, 1.0, 1.0, 1.0]
            }
        )

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
        refined_right_bboxes_nobkg_decoded = bbox_coder.decode(right_priors_nobkg, refined_right_bboxes_nobkg.view(-1, 4))
        refined_sbboxes_nobkg[:, 4:] = refined_right_bboxes_nobkg_decoded
        # select best keypoints
        right_keypts_pred_nobkg = right_keypts_pred_nobkg[
            torch.arange(mask_nonbackground.sum()),
            class_labels_pred_nobkg
        ]
        for indexKeypt in range(max_num_keypoints):
            right_keypts_pred_nobkg[:, (3 * indexKeypt):(2 + 3 * indexKeypt)] = DecodeKeypts(
                right_priors_nobkg,
                right_keypts_pred_nobkg[:, (3 * indexKeypt):(2 + 3 * indexKeypt)],
            )
        return mask_nonbackground, refined_sbboxes_nobkg, refined_right_scores_pred, right_keypts_pred_nobkg.view(mask_nonbackground.sum(), -1, 3)
