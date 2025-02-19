import os.path
import numpy
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
    lossRecords["BestIndex"].update(loss.item(), batchSize)
    lossRecords["Loss"].update(loss.item(), batchSize)

    if scaler is not None:
        scaler.scale(loss).backward()
        if clip_max_norm > 0:
            for key, suboptimizer in optimizer.items():
                scaler.unscale_(suboptimizer)
            for model in models.values():
                torch.nn.utils.clip_grad_norm_(model.parameters, clip_max_norm)
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

        with torch.autograd.set_detect_anomaly(True):
            # ---------- concentration net ----------
            left_event_sharp = _forward_one_batch(
                models["concentration_net"],
                {"x": batch_data["event"]["left"]},
                None,
                lossDictAll,
                {}
            )[0]
            right_event_sharp = _forward_one_batch(
                models["concentration_net"],
                {"x": batch_data["event"]["right"]},
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

            if models["disp_head"].module.is_freeze:
                # ---------- detr net ----------
                gt_labels_forrtdetr = [
                    {
                        "bboxes": onegt["bboxes"][..., :4].clone(),  # Note: only left bbox is needed for detr
                        "labels": onegt["labels"].clone()
                    } for onegt in batch_data["gt_labels"]["objdet"]
                ]
                global_step = epoch * len(data_loader) + indexBatch
                epoch_info = dict(epoch=epoch, step=indexBatch, global_step=global_step)
                left_detections, lossDictAll, artifacts = _forward_one_batch(
                    models["rtdetr"],
                    {
                        "x": batch_data["event"]["left"],
                        "x_right": batch_data["event"]["right"],
                    },
                    gt_labels_forrtdetr,
                    lossDictAll,
                    epoch_info,
                    scaler if not models["rtdetr"].module.is_freeze else None
                )
                right_feature, selected_leftdetections, corresponding_gt_labels, indices = artifacts

                if models["rtdetr"].module.is_freeze:
                    # prepare GT labels for stereo and keypts prediction:
                    batch_left_bboxes, batch_left_classes, batch_gt_labels_stereo, batch_gt_labels_keypt1, batch_gt_labels_keypt2 = [], [], [], [], []
                    for indexInBatch, one_left_bboxes in enumerate(selected_leftdetections):
                        one_left_bboxes_xyxy = torchvision.ops.box_convert(one_left_bboxes["bboxes"], in_fmt="cxcywh", out_fmt="xyxy".lower())
                        one_left_bboxes_xyxy[:, [0, 2]] *= imageWidth
                        one_left_bboxes_xyxy[:, [1, 3]] *= imageHeight
                        one_gt_bboxes_stereo = batch_data["gt_labels"]["objdet"][indexInBatch]["bboxes"]
                        one_gt_bboxes_stereo_reordered = one_gt_bboxes_stereo[indices[indexInBatch][1]]
                        one_gt_classes_stereo = batch_data["gt_labels"]["objdet"][indexInBatch]["labels"]
                        one_gt_classes_stereo_reordered = one_gt_classes_stereo[indices[indexInBatch][1]]
                        one_gt_featmap_keypt1 = batch_data["gt_labels"]["objdet"][indexInBatch]["keypt1_masks"]
                        one_gt_featmap_keypt1_reordered = one_gt_featmap_keypt1[indices[indexInBatch][1]]
                        one_gt_featmap_keypt2 = batch_data["gt_labels"]["objdet"][indexInBatch]["keypt2_masks"]
                        one_gt_featmap_keypt2_reordered = one_gt_featmap_keypt2[indices[indexInBatch][1]]
                        batch_left_bboxes.append(one_left_bboxes_xyxy)
                        batch_left_classes.append(one_left_bboxes["classes"])
                        batch_gt_labels_stereo.append({
                            "bboxes": one_gt_bboxes_stereo_reordered,
                            "classes": one_gt_classes_stereo_reordered
                        })
                        batch_gt_labels_keypt1.append(one_gt_featmap_keypt1_reordered)
                        batch_gt_labels_keypt2.append(one_gt_featmap_keypt2_reordered)

                    # ---------- stereo detection head ----------
                    stereo_preds, lossDictAll = _forward_one_batch(
                        models["stereo_detection_head"],
                        {
                            "right_feat": right_feature,
                            "left_bboxes": batch_left_bboxes,
                            "disp_prior": pred_disparity_pyramid[-1],
                            "batch_img_metas": {"h": imageHeight, "w": imageWidth},
                        },
                        batch_gt_labels_stereo,
                        lossDictAll,
                        {}
                    )[:2]

                    keypt1_preds, lossDictAll = _forward_one_batch(
                        models["featuremap_head"],
                        {
                            "img_feats": batch_data["event"]["left"],
                            "bbox_preds": batch_left_bboxes,
                            "class_preds": batch_left_classes,
                            "feature_id": "keypt1"
                        },
                        batch_gt_labels_keypt1,
                        lossDictAll,
                        {}
                    )[:2]

                    keypt2_preds, lossDictAll = _forward_one_batch(
                        models["featuremap_head"],
                        {
                            "img_feats": batch_data["event"]["left"],
                            "bbox_preds": batch_left_bboxes,
                            "class_preds": batch_left_classes,
                            "feature_id": "keypt2"
                        },
                        batch_gt_labels_keypt2,
                        lossDictAll,
                        {}
                    )[:2]

            # backward and optimize
            batchSize = batch_data["event"]["left"].shape[0]
            _backward_and_optimize(
                models,
                lossDictAll,
                optimizer,  # dict containing sub optimzers
                log_dict,
                batchSize,
                clip_max_norm,  # param used for amp
                scaler if not models["rtdetr"].module.is_freeze else None  # Note: only training detr we need this.
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

    log_dict = GetLogDict(is_train=False, is_secff=(hasattr(models["disp_head"].module, 'is_freeze') and not models["disp_head"].module.is_freeze))

    if tensorBoardLogger is not None:
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


        # ---------- concentration net ----------
        left_event_sharp = _forward_one_batch(
            models["concentration_net"],
            {"x": batch_data["event"]["left"]},
            None,
            lossDictAll,
            {}
        )[0]
        right_event_sharp = _forward_one_batch(
            models["concentration_net"],
            {"x": batch_data["event"]["right"]},
            None,
            lossDictAll,
            {}
        )[0]

        imageHeight, imageWidth = left_event_sharp.shape[-2:]

        # ---------- disp pred net ----------
        pred_disparity_pyramid, lossDictAll = _forward_one_batch(
            models["disp_head"],
            {
                "left_img": left_event_sharp,
                "right_img": right_event_sharp
            },
            batch_data["gt_labels"]["disparity"],
            lossDictAll,
            {}
        )[:2]

        if models["disp_head"].module.is_freeze:
            # ---------- detr net ----------
            gt_labels_forrtdetr = [
                {
                    "bboxes": onegt["bboxes"][..., :4].clone(),  # Note: only left bbox is needed for detr
                    "labels": onegt["labels"].clone()
                } for onegt in batch_data["gt_labels"]["objdet"]
            ]
            global_step = epoch * len(data_loader) + indexBatch
            epoch_info = dict(epoch=epoch, step=indexBatch, global_step=global_step)
            left_detections, lossDictAll, artifacts = _forward_one_batch(
                models["rtdetr"],
                {
                    "x": batch_data["event"]["left"],
                    "x_right": batch_data["event"]["right"],
                },
                gt_labels_forrtdetr,
                lossDictAll,
                epoch_info,
            )
            right_feature, selected_leftdetections, corresponding_gt_labels, indices = artifacts

            if models["rtdetr"].module.is_freeze:
                # prepare GT labels for stereo and keypts prediction:
                batch_left_bboxes, batch_left_classes, batch_gt_labels_stereo, batch_gt_labels_keypt1, batch_gt_labels_keypt2 = [], [], [], [], []
                for indexInBatch, one_left_bboxes in enumerate(selected_leftdetections):
                    one_left_bboxes_xyxy = torchvision.ops.box_convert(one_left_bboxes["bboxes"], in_fmt="cxcywh", out_fmt="xyxy".lower())
                    one_left_bboxes_xyxy[:, [0, 2]] *= imageWidth
                    one_left_bboxes_xyxy[:, [1, 3]] *= imageHeight
                    one_gt_bboxes_stereo = batch_data["gt_labels"]["objdet"][indexInBatch]["bboxes"]
                    one_gt_bboxes_stereo_reordered = one_gt_bboxes_stereo[indices[indexInBatch][1]]
                    one_gt_classes_stereo = batch_data["gt_labels"]["objdet"][indexInBatch]["labels"]
                    one_gt_classes_stereo_reordered = one_gt_classes_stereo[indices[indexInBatch][1]]
                    one_gt_featmap_keypt1 = batch_data["gt_labels"]["objdet"][indexInBatch]["keypt1_masks"]
                    one_gt_featmap_keypt1_reordered = one_gt_featmap_keypt1[indices[indexInBatch][1]]
                    one_gt_featmap_keypt2 = batch_data["gt_labels"]["objdet"][indexInBatch]["keypt2_masks"]
                    one_gt_featmap_keypt2_reordered = one_gt_featmap_keypt2[indices[indexInBatch][1]]
                    batch_left_bboxes.append(one_left_bboxes_xyxy)
                    batch_left_classes.append(one_left_bboxes["classes"])
                    batch_gt_labels_stereo.append({
                        "bboxes": one_gt_bboxes_stereo_reordered,
                        "classes": one_gt_classes_stereo_reordered
                    })
                    batch_gt_labels_keypt1.append(one_gt_featmap_keypt1_reordered)
                    batch_gt_labels_keypt2.append(one_gt_featmap_keypt2_reordered)

                # ---------- stereo detection head ----------
                stereo_preds, lossDictAll = _forward_one_batch(
                    models["stereo_detection_head"],
                    {
                        "right_feat": right_feature,
                        "left_bboxes": batch_left_bboxes,
                        "disp_prior": pred_disparity_pyramid[-1],
                        "batch_img_metas": {"h": imageHeight, "w": imageWidth},
                    },
                    batch_gt_labels_stereo,
                    lossDictAll,
                    {}
                )[:2]

                keypt1_preds, lossDictAll = _forward_one_batch(
                    models["featuremap_head"],
                    {
                        "img_feats": batch_data["event"]["left"],
                        "bbox_preds": batch_left_bboxes,
                        "class_preds": batch_left_classes,
                        "feature_id": "keypt1"
                    },
                    batch_gt_labels_keypt1,
                    lossDictAll,
                    {}
                )[:2]

                keypt2_preds, lossDictAll = _forward_one_batch(
                    models["featuremap_head"],
                    {
                        "img_feats": batch_data["event"]["left"],
                        "bbox_preds": batch_left_bboxes,
                        "class_preds": batch_left_classes,
                        "feature_id": "keypt2"
                    },
                    batch_gt_labels_keypt2,
                    lossDictAll,
                    {}
                )[:2]

        if hasattr(models["disp_head"].module, 'is_freeze') and not models["disp_head"].module.is_freeze:
            log_dict["EPE"].update(pred_disparity_pyramid[-1].cpu(), batch_data["disparity"].cpu(), mask.cpu())
            log_dict["1PE"].update(pred_disparity_pyramid[-1].cpu(), batch_data["disparity"].cpu(), mask.cpu())
            log_dict["2PE"].update(pred_disparity_pyramid[-1].cpu(), batch_data["disparity"].cpu(), mask.cpu())
            log_dict["RMSE"].update(pred_disparity_pyramid[-1].cpu(), batch_data["disparity"].cpu(), mask.cpu())

        if tensorBoardLogger is not None:
            pbar.update(1)

    torch.cuda.synchronize()

    if logger is not None:
        pbar.close()

    return log_dict