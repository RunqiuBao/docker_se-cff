import os.path
import numpy
import torch.nn.functional as F
import torch
import torch.distributed as dist
import cv2
import time
import torchvision
from typing import Optional

from tqdm import tqdm
from collections import OrderedDict

from utils import visualizer
from .visz_utils import DrawResultBboxesAndKeyptsOnStereoEventFrame, RenderImageWithBboxes
from ..models.utils.misc import freeze_module_grads, convert_tensor_to_numpy, DetachCopyNested

from ..models.utils.objdet_utils import SelectTargets, SelectTopkCandidates
from ..methods.visz_utils import RenderImageWithBboxesAndKeypts
from .log_utils import GetLogDict
from .base import batch_to_cuda

from..models.utils.misc import freeze_module_grads

import logging
logger = logging.getLogger(__name__)


def freeze_static_components(models: dict):
    """
    freeze its gradients, if no need to train it.
    """
    for key, model in models.items():
        if model.module.is_freeze:
            logger.info("---- freeze params for {}".format(key))
            freeze_module_grads(models[key])


def _forward_one_batch(
    model: torch.nn.Module,
    model_inputs: dict,
    labels: Optional[torch.Tensor],
    lossDictAll: dict,
    necessary_info: dict, 
    scaler: Optional[torch.cuda.amp.grad_scaler.GradScaler] = None
):
    starttime = time.time()
    artifacts = None
    if scaler is not None:
        with torch.autocast(device_type="cuda", cache_enabled=True):
            preds, losses, artifacts = model(**model_inputs, labels=labels, **necessary_info)
    else:
        preds, losses, artifacts = model(**model_inputs, labels=labels, **necessary_info)
    if losses is not None:
        lossDictAll.update(losses)  # Note: losses[0] is a dict of all losses
    logger.debug("-> forward ({}) time cost: {}".format(type(model).__name__, time.time() - starttime))
    return preds, lossDictAll, artifacts


def _backward_and_optimize(
    models: dict,
    lossDictCurrentStep: dict,
    optimizer: dict,  # dict containing sub optimzers
    lossRecords: dict,
    batchSize: int,
    clip_max_norm: Optional[float] = None,  # param used for amp
    scaler: Optional[torch.cuda.amp.grad_scaler.GradScaler] = None,
):
    starttime = time.time()
    loss = 0
    for key, value in lossDictCurrentStep.items():
        loss += value
        if key in lossRecords:
            lossRecords[key].update(lossDictCurrentStep[key].item(), batchSize)
    lossRecords["BestIndex"].update(loss.item() if loss != 0 else 0, batchSize)
    lossRecords["Loss"].update(loss.item() if loss != 0 else 0, batchSize)

    if scaler is not None:
        scaler.scale(loss).backward()
        if clip_max_norm > 0:
            for key, suboptimizer in optimizer.items():
                scaler.unscale_(suboptimizer)
            for model in models.values():
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        for key, suboptimizer in optimizer.items():
            if not models[key].module.is_freeze:
                scaler.step(suboptimizer)
        scaler.update()
    else:
        loss.backward()  # Note: PyTorch’s autograd engine ensures that gradients are only computed for parameters that contribute to a given loss term.
        for key, suboptimizer in optimizer.items():
            if not models[key].module.is_freeze:
                suboptimizer.step()
    logger.debug("-> backward_and_optimize time cost: {}".format(time.time() - starttime))
    return


def train(
    models,  # dict of models used in stereoeventobjectdetection task
    data_loader,
    optimizer,
    tensorBoardLogger,
    scaler=None,  # Note: GradScaler used for auto mixed precision
    ema=None,
    clip_max_norm=None,
    is_distributed=False,
    world_size=1,
    epoch=None
):
    """
    Args:
        ...
        ema: Exponential Moving Average. Smoothing between epoches.
    """
    for model in models.values():
        model.train()

    log_dict = GetLogDict(is_train=True, is_secff=(hasattr(models["disp_head"].module, 'is_freeze') and not models["disp_head"].module.is_freeze))
    lossDictAll = {}

    pbar = tqdm(total=len(data_loader))
    data_iter = iter(data_loader)
    for indexBatch in range(len(data_loader)):
        batch_data = next(data_iter)
        if hasattr(models["disp_head"].module, 'is_freeze') and not models["disp_head"].module.is_freeze:
            mask = batch_data["gt_labels"]["disparity"] > 0
            if not mask.any():
                continue
        if "bboxes" not in batch_data["gt_labels"]["objdet"][0]:
            print("Error: the batch data do not contain GT for bboxes.")
            continue
        batch_data = batch_to_cuda(batch_data)
        # classes labels in objdet need to be int
        for indexObj in range(len(batch_data['gt_labels']['objdet'])):
            batch_data['gt_labels']['objdet'][indexObj]['labels'] = batch_data['gt_labels']['objdet'][indexObj]['labels'].to(torch.long)
            batch_data['objdet'][indexObj]['labels'] = batch_data['objdet'][indexObj]['labels'].to(torch.long)

        for key, suboptimizer in optimizer.items():
            if not models[key].module.is_freeze:
                suboptimizer.zero_grad()

        # ---------- concentration net ----------
        (left_event_sharp, right_event_sharp) = _forward_one_batch(
            models["concentration_net"],
            {"left_img": batch_data["event"]["left"], "right_img": batch_data["event"]["right"]},
            None,
            lossDictAll,
            {}
        )[0]

        imageHeight, imageWidth = batch_data["event"]["left"].shape[-2:]

        # ---------- disp pred net ----------
        pred_disparity_pyramid, lossDictAll = _forward_one_batch(
            models["disp_head"],
            {"left_img": left_event_sharp, "right_img": right_event_sharp},
            batch_data["gt_labels"]["disparity"],
            lossDictAll,
            {}
        )[:2]

        # @@@@@@@@@@@@@@@@@@@@ VISUALIZATION @@@@@@@@@@@@@@@@@@@@
        if tensorBoardLogger is not None:
            disp_map = pred_disparity_pyramid[-1].detach().cpu()
            disp_map *= 255 / disp_map.max()
            tensorBoardLogger.add_image("disp_map", disp_map.to(torch.uint8).squeeze())

        if models["disp_head"].module.is_freeze:
            # ---------- objdet net ----------
            left_detections, lossDictAll, artifacts = _forward_one_batch(
                models["objdet_head"],
                {
                    "left_event_voxel": batch_data["event"]["left"],
                    "right_event_voxel": batch_data["event"]["right"],
                    "batch_img_metas": {"h": imageHeight, "w": imageWidth},
                },
                batch_data["gt_labels"]["objdet"],
                lossDictAll,
                {},
            )
            (
                right_feature,
                left_selected_boxes,  # Note: only use this when training left objdet
                left_selected_classes,
                left_selected_confidences
            ) = artifacts

            # @@@@@@@@@@@@@@@@@@@@ VISUALIZATION @@@@@@@@@@@@@@@@@@@@
            if tensorBoardLogger is not None and left_selected_boxes is not None:
                leftimage_visz = RenderImageWithBboxes(
                    left_event_sharp.detach().squeeze(1).cpu().numpy(),
                    {
                        "bboxes": left_selected_boxes[0],
                        "classes": left_selected_classes[0],
                    }
                )
                tensorBoardLogger.add_image("(train) left sharp with bboxes", leftimage_visz[0])

            if models["objdet_head"].module.is_freeze:
                (
                    left_bboxes,
                    left_cls_scores,
                    left_objectnesses
                ) = left_detections
                
                preds_items_multilevels_detachcopy = DetachCopyNested([left_cls_scores, left_bboxes, left_objectnesses])
                batch_img_metas = {"h": imageHeight, "w": imageWidth}

                (
                    left_cls_scores_selected,  # shape (B, 100, num_clqss)
                    left_bboxes_selected,  # shape is (B, 100, 4). [tl_x, tl_y, br_x, br_y] format bbox, all in global scale
                    left_objectness_selected,  # Note: shape is [B, 100,]
                    priors_selected
                ) = SelectTopkCandidates(
                    *preds_items_multilevels_detachcopy,
                    img_metas=batch_img_metas,
                    config={
                        "strides": models["objdet_head"].module.config["strides"],
                        "freeze_leftobjdet": models["objdet_head"].module.config["is_freeze"],
                        "nms_pre": models["objdet_head"].module.config["nms_pre"],
                        "num_classes": models["objdet_head"].module.config["num_classes"],
                        "min_bbox_size": models["objdet_head"].module.config["min_bbox_size"],
                        "nms_iou_threshold": models["objdet_head"].module.config["nms_iou_threshold"],
                        "num_topk_candidates": models["objdet_head"].module.config["num_topk_candidates"]
                    }
                )

                (
                    num_pos,
                    pos_masks,
                    cls_targets,
                    bbox_targets,
                    indices_bbox_targets,
                    batch_num_pos_per_img
                ) = SelectTargets(
                    priors_selected,
                    left_cls_scores_selected,
                    left_bboxes_selected,
                    left_objectness_selected,
                    batch_data["gt_labels"]["objdet"],
                    config={
                        "num_classes": models["objdet_head"].module.config["num_classes"],
                    }
                )

                if num_pos > 0:
                    # ---------- stereo detection head ----------
                    stereo_preds, lossDictAll, artifacts = _forward_one_batch(
                        models["stereo_detection_head"],
                        {
                            "right_feat": right_feature,
                            "left_bboxes": left_bboxes_selected,
                            "disp_prior": pred_disparity_pyramid[-1],
                            "batch_img_metas": batch_img_metas,
                            "detector_format": "yolox"
                        },
                        {
                            "pos_masks": pos_masks,
                            "cls_targets": cls_targets,
                            "bbox_targets": bbox_targets,
                            "indices_bbox_targets": indices_bbox_targets,
                            "batch_num_pos_per_img": batch_num_pos_per_img
                        },
                        lossDictAll,
                        {}
                    )

                    # @@@@@@@@@@@@@@@@@@@@ VISUALIZATION @@@@@@@@@@@@@@@@@@@@
                    if tensorBoardLogger is not None:
                        right_bboxes = artifacts[0].detach()[0]
                        right_bboxes = torch.cat([
                            right_bboxes[..., 4].unsqueeze(-1),
                            right_bboxes[..., 1].unsqueeze(-1),
                            right_bboxes[..., 5].unsqueeze(-1),
                            right_bboxes[..., 3].unsqueeze(-1)
                        ], dim=-1)
                        rightimage_visz = RenderImageWithBboxes(
                            right_event_sharp[0].detach().squeeze(1).cpu().numpy(),
                            {
                                "bboxes": right_bboxes[pos_masks],
                                "classes": torch.max(left_cls_scores_selected[0][pos_masks], dim=-1)[-1],
                            }
                        )
                        tensorBoardLogger.add_image("(train) right sharp with bboxes", rightimage_visz[0])

                    batch_left_bboxes, batch_left_class_preds, batch_gt_labels_keypt1, batch_gt_labels_keypt2 = [], [], [], []
                    batch_size, num_priors = left_bboxes_selected.shape[:2]
                    for indexInBatch in range(batch_size):
                        bbox_preds_oneimg = left_bboxes_selected[indexInBatch][pos_masks[indexInBatch * num_priors:(indexInBatch + 1) * num_priors]]
                        batch_left_bboxes.append(bbox_preds_oneimg)
                        class_preds_oneimg = torch.max(left_cls_scores_selected[indexInBatch][pos_masks[indexInBatch * num_priors:(indexInBatch + 1) * num_priors]])
                        batch_left_class_preds.append(class_preds_oneimg)

                        one_gt_featmap_keypt1 = batch_data["gt_labels"]["objdet"][indexInBatch]["keypt1_masks"]
                        num_pos_until_this_img = sum(batch_num_pos_per_img[:indexInBatch])
                        num_pos_after_this_img = sum(batch_num_pos_per_img[:indexInBatch + 1])
                        one_gt_featmap_keypt1_reordered = one_gt_featmap_keypt1[indices_bbox_targets.squeeze(-1)[num_pos_until_this_img:num_pos_after_this_img].to(torch.int)]
                        batch_gt_labels_keypt1.append(one_gt_featmap_keypt1_reordered)

                        one_gt_featmap_keypt2 = batch_data["gt_labels"]["objdet"][indexInBatch]["keypt2_masks"]
                        one_gt_featmap_keypt2_reordered = one_gt_featmap_keypt2[indices_bbox_targets.squeeze(-1)[num_pos_until_this_img:num_pos_after_this_img].to(torch.int)]
                        batch_gt_labels_keypt2.append(one_gt_featmap_keypt2_reordered)

                    keypt1_preds, lossDictAll, artifacts = _forward_one_batch(
                        models["featuremap_head"],
                        {
                            "img_feats": batch_data["event"]["left"],
                            "bbox_preds": batch_left_bboxes,
                            "class_preds": batch_left_class_preds,
                            "feature_id": "keypt1"
                        },
                        batch_gt_labels_keypt1,
                        lossDictAll,
                        {}
                    )

                    # @@@@@@@@@@@@@@@@@@@@ VISUALIZATION @@@@@@@@@@@@@@@@@@@@
                    if tensorBoardLogger is not None:
                        if artifacts is not None:
                            list_featmap_preds_inclass, list_mask_targets = artifacts
                            onefeatmap = list_featmap_preds_inclass[0][0].detach().cpu()
                            onefeatmap -= onefeatmap.min()
                            onefeatmap *= 255 / onefeatmap.max()
                            onefeatmap = F.interpolate(onefeatmap.unsqueeze(0).unsqueeze(0), size=(720, 720), mode='bilinear', align_corners=False)
                            tensorBoardLogger.add_image("keypt1_prediction_sample", onefeatmap.to(torch.uint8).squeeze())
                            featmap_target = list_mask_targets[0][0].detach().cpu()
                            featmap_target -= featmap_target.min()
                            featmap_target *= 255 / featmap_target.max()
                            featmap_target = F.interpolate(featmap_target.unsqueeze(0).unsqueeze(0), size=(720, 720), mode='bilinear', align_corners=False)
                            tensorBoardLogger.add_image("keypt1_target_sample", featmap_target.to(torch.uint8).squeeze())

                    keypt2_preds, lossDictAll, artifacts = _forward_one_batch(
                        models["featuremap_head"],
                        {
                            "img_feats": batch_data["event"]["left"],
                            "bbox_preds": batch_left_bboxes,
                            "class_preds": batch_left_class_preds,
                            "feature_id": "keypt2"
                        },
                        batch_gt_labels_keypt2,
                        lossDictAll,
                        {}
                    )

                    # @@@@@@@@@@@@@@@@@@@@ VISUALIZATION @@@@@@@@@@@@@@@@@@@@
                    if tensorBoardLogger is not None:
                        if artifacts is not None:
                            list_featmap_preds_inclass, list_mask_targets = artifacts
                            onefeatmap = list_featmap_preds_inclass[0][0].detach().cpu()
                            onefeatmap -= onefeatmap.min()
                            onefeatmap *= 255 / onefeatmap.max()
                            onefeatmap = F.interpolate(onefeatmap.unsqueeze(0).unsqueeze(0), size=(720, 720), mode='bilinear', align_corners=False)
                            tensorBoardLogger.add_image("keypt2_prediction_sample", onefeatmap.to(torch.uint8).squeeze())
                            featmap_target = list_mask_targets[0][0].detach().cpu()
                            featmap_target -= featmap_target.min()
                            featmap_target *= 255 / featmap_target.max()
                            featmap_target = F.interpolate(featmap_target.unsqueeze(0).unsqueeze(0), size=(720, 720), mode='bilinear', align_corners=False)
                            tensorBoardLogger.add_image("keypt2_target_sample", featmap_target.to(torch.uint8).squeeze())

        # backward and optimize
        batchSize = batch_data["event"]["left"].shape[0]
        _backward_and_optimize(
            models,
            lossDictAll,
            optimizer,  # dict containing sub optimzers
            log_dict,
            batchSize
        )

        if ema is not None:
            # exponential moving average
            for key, model in models.items():
                if not model.module.is_freeze:
                    ema[key].update(model)

        if hasattr(models["disp_head"].module, 'is_freeze') and not models["disp_head"].module.is_freeze:
            log_dict["EPE"].update(pred_disparity_pyramid[-1].cpu(), batch_data["disparity"].cpu(), mask.cpu())
            log_dict["1PE"].update(pred_disparity_pyramid[-1].cpu(), batch_data["disparity"].cpu(), mask.cpu())
            log_dict["2PE"].update(pred_disparity_pyramid[-1].cpu(), batch_data["disparity"].cpu(), mask.cpu())
            log_dict["RMSE"].update(pred_disparity_pyramid[-1].cpu(), batch_data["disparity"].cpu(), mask.cpu())

        if tensorBoardLogger is not None:
            pbar.update(1)
        torch.cuda.synchronize()

    if tensorBoardLogger is not None:
        pbar.close()
    return log_dict


@torch.no_grad()
def valid(
    models,
    data_loader,
    is_distributed=False,
    world_size=1,
    tensorBoardLogger=None,
    epoch=None
):
    """
    Args:
        ...
        ema: Exponential Moving Average. Smoothing between epoches.
    """
    for model in models.values():
        model.eval()

    log_dict = GetLogDict(is_train=False, is_secff=(hasattr(models["disp_head"], 'is_freeze') and not models["disp_head"].is_freeze))
    lossDictAll = {}

    if tensorBoardLogger is not None:
        pbar = tqdm(total=len(data_loader))
    data_iter = iter(data_loader)
    for indexBatch in range(len(data_loader)):
        batch_data = next(data_iter)
        if hasattr(models["disp_head"], 'is_freeze') and not models["disp_head"].is_freeze:
            mask = batch_data["gt_labels"]["disparity"] > 0
            if not mask.any():
                continue
        if "bboxes" not in batch_data["gt_labels"]["objdet"][0]:
            print("Error: the batch data do not contain GT for bboxes.")
            continue
        batch_data = batch_to_cuda(batch_data)
        # classes labels in objdet need to be int
        for indexObj in range(len(batch_data['gt_labels']['objdet'])):
            batch_data['gt_labels']['objdet'][indexObj]['labels'] = batch_data['gt_labels']['objdet'][indexObj]['labels'].to(torch.long)
            batch_data['objdet'][indexObj]['labels'] = batch_data['objdet'][indexObj]['labels'].to(torch.long)

        # ---------- concentration net ----------
        (left_event_sharp, right_event_sharp) = _forward_one_batch(
            models["concentration_net"],
            {"left_img": batch_data["event"]["left"], "right_img": batch_data["event"]["right"]},
            None,
            lossDictAll,
            {}
        )[0]

        imageHeight, imageWidth = batch_data["event"]["left"].shape[-2:]

        # ---------- disp pred net ----------
        pred_disparity_pyramid, lossDictAll = _forward_one_batch(
            models["disp_head"],
            {"left_img": left_event_sharp, "right_img": right_event_sharp},
            batch_data["gt_labels"]["disparity"],
            lossDictAll,
            {}
        )[:2]

        # @@@@@@@@@@@@@@@@@@@@ VISUALIZATION @@@@@@@@@@@@@@@@@@@@
        if tensorBoardLogger is not None:
            disp_map = pred_disparity_pyramid[-1].detach().cpu()
            disp_map *= 255 / disp_map.max()
            tensorBoardLogger.add_image("disp_map", disp_map.to(torch.uint8).squeeze())

        if models["disp_head"].is_freeze:
            # ---------- objdet net ----------
            left_detections, lossDictAll, artifacts = _forward_one_batch(
                models["objdet_head"],
                {
                    "left_event_voxel": batch_data["event"]["left"],
                    "right_event_voxel": batch_data["event"]["right"],
                    "batch_img_metas": {"h": imageHeight, "w": imageWidth},
                },
                batch_data["gt_labels"]["objdet"],
                lossDictAll,
                {},
            )
            (
                right_feature,
                left_selected_boxes,  # Note: only use this when training left objdet
                left_selected_classes,
                left_selected_confidences
            ) = artifacts

            # @@@@@@@@@@@@@@@@@@@@ VISUALIZATION @@@@@@@@@@@@@@@@@@@@
            if tensorBoardLogger is not None and left_selected_boxes is not None:
                leftimage_visz = RenderImageWithBboxes(
                    left_event_sharp.detach().squeeze(1).cpu().numpy(),
                    {
                        "bboxes": left_selected_boxes[0],
                        "classes": left_selected_classes[0],
                    }
                )
                tensorBoardLogger.add_image("(train) left sharp with bboxes", leftimage_visz[0])

            if models["objdet_head"].is_freeze:
                (
                    left_bboxes,
                    left_cls_scores,
                    left_objectnesses
                ) = left_detections
                
                preds_items_multilevels_detachcopy = DetachCopyNested([left_cls_scores, left_bboxes, left_objectnesses])
                batch_img_metas = {"h": imageHeight, "w": imageWidth}

                (
                    left_cls_scores_selected,  # shape (B, 100, num_clqss)
                    left_bboxes_selected,  # shape is (B, 100, 4). [tl_x, tl_y, br_x, br_y] format bbox, all in global scale
                    left_objectness_selected,  # Note: shape is [B, 100,]
                    priors_selected
                ) = SelectTopkCandidates(
                    *preds_items_multilevels_detachcopy,
                    img_metas=batch_img_metas,
                    config={
                        "strides": models["objdet_head"].config["strides"],
                        "freeze_leftobjdet": models["objdet_head"].config["is_freeze"],
                        "nms_pre": models["objdet_head"].config["nms_pre"],
                        "num_classes": models["objdet_head"].config["num_classes"],
                        "min_bbox_size": models["objdet_head"].config["min_bbox_size"],
                        "nms_iou_threshold": models["objdet_head"].config["nms_iou_threshold"],
                        "num_topk_candidates": models["objdet_head"].config["num_topk_candidates"]
                    }
                )

                (
                    num_pos,
                    pos_masks,
                    cls_targets,
                    bbox_targets,
                    indices_bbox_targets,
                    batch_num_pos_per_img
                ) = SelectTargets(
                    priors_selected,
                    left_cls_scores_selected,
                    left_bboxes_selected,
                    left_objectness_selected,
                    batch_data["gt_labels"]["objdet"],
                    config={
                        "num_classes": models["objdet_head"].config["num_classes"],
                    }
                )

                if num_pos > 0:
                    # ---------- stereo detection head ----------
                    stereo_preds, lossDictAll, artifacts = _forward_one_batch(
                        models["stereo_detection_head"],
                        {
                            "right_feat": right_feature,
                            "left_bboxes": left_bboxes_selected,
                            "disp_prior": pred_disparity_pyramid[-1],
                            "batch_img_metas": batch_img_metas,
                            "detector_format": "yolox"
                        },
                        {
                            "pos_masks": pos_masks,
                            "cls_targets": cls_targets,
                            "bbox_targets": bbox_targets,
                            "indices_bbox_targets": indices_bbox_targets,
                            "batch_num_pos_per_img": batch_num_pos_per_img
                        },
                        lossDictAll,
                        {}
                    )
                    from IPython import embed; embed()

                    # @@@@@@@@@@@@@@@@@@@@ VISUALIZATION @@@@@@@@@@@@@@@@@@@@
                    if tensorBoardLogger is not None:
                        right_bboxes = artifacts[0].detach()[0]
                        right_bboxes = torch.cat([
                            right_bboxes[..., 4].unsqueeze(-1),
                            right_bboxes[..., 1].unsqueeze(-1),
                            right_bboxes[..., 5].unsqueeze(-1),
                            right_bboxes[..., 3].unsqueeze(-1)
                        ], dim=-1)
                        rightimage_visz = RenderImageWithBboxes(
                            right_event_sharp[0].detach().squeeze(1).cpu().numpy(),
                            {
                                "bboxes": right_bboxes[pos_masks],
                                "classes": torch.max(left_cls_scores_selected[0][pos_masks], dim=-1)[-1],
                            }
                        )
                        tensorBoardLogger.add_image("(train) right sharp with bboxes", rightimage_visz[0])

                    batch_left_bboxes, batch_left_class_preds, batch_gt_labels_keypt1, batch_gt_labels_keypt2 = [], [], [], []
                    batch_size, num_priors = left_bboxes_selected.shape[:2]
                    for indexInBatch in range(batch_size):
                        bbox_preds_oneimg = left_bboxes_selected[indexInBatch][pos_masks[indexInBatch * num_priors:(indexInBatch + 1) * num_priors]]
                        batch_left_bboxes.append(bbox_preds_oneimg)
                        class_preds_oneimg = torch.max(left_cls_scores_selected[indexInBatch][pos_masks[indexInBatch * num_priors:(indexInBatch + 1) * num_priors]])
                        batch_left_class_preds.append(class_preds_oneimg)

                        one_gt_featmap_keypt1 = batch_data["gt_labels"]["objdet"][indexInBatch]["keypt1_masks"]
                        num_pos_until_this_img = sum(batch_num_pos_per_img[:indexInBatch])
                        num_pos_after_this_img = sum(batch_num_pos_per_img[:indexInBatch + 1])
                        one_gt_featmap_keypt1_reordered = one_gt_featmap_keypt1[indices_bbox_targets.squeeze(-1)[num_pos_until_this_img:num_pos_after_this_img].to(torch.int)]
                        batch_gt_labels_keypt1.append(one_gt_featmap_keypt1_reordered)

                        one_gt_featmap_keypt2 = batch_data["gt_labels"]["objdet"][indexInBatch]["keypt2_masks"]
                        one_gt_featmap_keypt2_reordered = one_gt_featmap_keypt2[indices_bbox_targets.squeeze(-1)[num_pos_until_this_img:num_pos_after_this_img].to(torch.int)]
                        batch_gt_labels_keypt2.append(one_gt_featmap_keypt2_reordered)

                    keypt1_preds, lossDictAll, artifacts = _forward_one_batch(
                        models["featuremap_head"],
                        {
                            "img_feats": batch_data["event"]["left"],
                            "bbox_preds": batch_left_bboxes,
                            "class_preds": batch_left_class_preds,
                            "feature_id": "keypt1"
                        },
                        batch_gt_labels_keypt1,
                        lossDictAll,
                        {}
                    )

                    # @@@@@@@@@@@@@@@@@@@@ VISUALIZATION @@@@@@@@@@@@@@@@@@@@
                    if tensorBoardLogger is not None:
                        if artifacts is not None:
                            list_featmap_preds_inclass, list_mask_targets = artifacts
                            onefeatmap = list_featmap_preds_inclass[0][0].detach().cpu()
                            onefeatmap -= onefeatmap.min()
                            onefeatmap *= 255 / onefeatmap.max()
                            onefeatmap = F.interpolate(onefeatmap.unsqueeze(0).unsqueeze(0), size=(720, 720), mode='bilinear', align_corners=False)
                            tensorBoardLogger.add_image("keypt1_prediction_sample", onefeatmap.to(torch.uint8).squeeze())
                            featmap_target = list_mask_targets[0][0].detach().cpu()
                            featmap_target -= featmap_target.min()
                            featmap_target *= 255 / featmap_target.max()
                            featmap_target = F.interpolate(featmap_target.unsqueeze(0).unsqueeze(0), size=(720, 720), mode='bilinear', align_corners=False)
                            tensorBoardLogger.add_image("keypt1_target_sample", featmap_target.to(torch.uint8).squeeze())

                    keypt2_preds, lossDictAll, artifacts = _forward_one_batch(
                        models["featuremap_head"],
                        {
                            "img_feats": batch_data["event"]["left"],
                            "bbox_preds": batch_left_bboxes,
                            "class_preds": batch_left_class_preds,
                            "feature_id": "keypt2"
                        },
                        batch_gt_labels_keypt2,
                        lossDictAll,
                        {}
                    )

                    # @@@@@@@@@@@@@@@@@@@@ VISUALIZATION @@@@@@@@@@@@@@@@@@@@
                    if tensorBoardLogger is not None:
                        if artifacts is not None:
                            list_featmap_preds_inclass, list_mask_targets = artifacts
                            onefeatmap = list_featmap_preds_inclass[0][0].detach().cpu()
                            onefeatmap -= onefeatmap.min()
                            onefeatmap *= 255 / onefeatmap.max()
                            onefeatmap = F.interpolate(onefeatmap.unsqueeze(0).unsqueeze(0), size=(720, 720), mode='bilinear', align_corners=False)
                            tensorBoardLogger.add_image("keypt2_prediction_sample", onefeatmap.to(torch.uint8).squeeze())
                            featmap_target = list_mask_targets[0][0].detach().cpu()
                            featmap_target -= featmap_target.min()
                            featmap_target *= 255 / featmap_target.max()
                            featmap_target = F.interpolate(featmap_target.unsqueeze(0).unsqueeze(0), size=(720, 720), mode='bilinear', align_corners=False)
                            tensorBoardLogger.add_image("keypt2_target_sample", featmap_target.to(torch.uint8).squeeze())

        batchSize = batch_data["event"]["left"].shape[0]
        loss = 0
        for key, value in lossDictAll.items():
            loss += value
            if key in log_dict:
                log_dict[key].update(lossDictAll[key].item(), batchSize)
        log_dict["BestIndex"].update(loss.item() if loss != 0 else 0, batchSize)
        log_dict["Loss"].update(loss.item() if loss != 0 else 0, batchSize)

        if hasattr(models["disp_head"], 'is_freeze') and not models["disp_head"].is_freeze:
            log_dict["EPE"].update(pred_disparity_pyramid[-1].cpu(), batch_data["disparity"].cpu(), mask.cpu())
            log_dict["1PE"].update(pred_disparity_pyramid[-1].cpu(), batch_data["disparity"].cpu(), mask.cpu())
            log_dict["2PE"].update(pred_disparity_pyramid[-1].cpu(), batch_data["disparity"].cpu(), mask.cpu())
            log_dict["RMSE"].update(pred_disparity_pyramid[-1].cpu(), batch_data["disparity"].cpu(), mask.cpu())

        if tensorBoardLogger is not None:
            pbar.update(1)

    torch.cuda.synchronize()

    if tensorBoardLogger is not None:
        pbar.close()

    return log_dict