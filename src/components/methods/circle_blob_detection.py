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
from ..models.utils.misc import freeze_module_grads, convert_tensor_to_numpy
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
    lossRecords["BestIndex"].update(loss.item(), batchSize)
    lossRecords["Loss"].update(loss.item(), batchSize)

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

    log_dict = GetLogDict()
    lossDictAll = {}

    pbar = tqdm(total=len(data_loader))
    data_iter = iter(data_loader)
    for indexBatch in range(len(data_loader)):
        batch_data = next(data_iter)

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

        imageHeight, imageWidth = batch_data["image"].shape[-2:]

        # ---------- detr net ----------
        gt_labels_forrtdetr = []
        for onegt in batch_data["gt_labels"]["objdet"]: 
            onegt = {
                "bboxes": torchvision.ops.box_convert(onegt["bboxes"][..., :4].clone(), in_fmt="xyxy", out_fmt="cxcywh"),  # Note: only left bbox is needed for detr
                "labels": onegt["labels"].clone()
            }
            onegt["bboxes"][:, [0, 2]] /= imageWidth
            onegt["bboxes"][:, [1, 3]] /= imageHeight
            gt_labels_forrtdetr.append(onegt)
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

        # @@@@@@@@@@@@@@@@@@@@ VISUALIZATION @@@@@@@@@@@@@@@@@@@@
        if tensorBoardLogger is not None:
            left_bboxes = selected_leftdetections[0]["bboxes"]
            left_bboxes = torchvision.ops.box_convert(left_bboxes.clone(), in_fmt="cxcywh", out_fmt="xyxy")
            left_bboxes[:, [0, 2]] *= imageWidth
            left_bboxes[:, [1, 3]] *= imageHeight
            leftimage_visz = RenderImageWithBboxes(
                left_event_sharp.detach().squeeze(1).cpu().numpy(),
                {
                    "bboxes": left_bboxes,
                    "classes": selected_leftdetections[0]["classes"],
                }
            )
            tensorBoardLogger.add_image("(train) left sharp with bboxes", leftimage_visz[0])

        # backward and optimize
        batchSize = batch_data["image"].shape[0]
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

    log_dict = GetLogDict()
    lossDictAll = {}

    if tensorBoardLogger is not None:
        pbar = tqdm(total=len(data_loader))
    data_iter = iter(data_loader)
    for indexBatch in range(len(data_loader)):
        batch_data = next(data_iter)

        if "bboxes" not in batch_data["gt_labels"]["objdet"][0]:
            print("Error: the batch data do not contain GT for bboxes.")
            continue

        batch_data = batch_to_cuda(batch_data)
        # classes labels in objdet need to be int
        for indexObj in range(len(batch_data['gt_labels']['objdet'])):
            batch_data['gt_labels']['objdet'][indexObj]['labels'] = batch_data['gt_labels']['objdet'][indexObj]['labels'].to(torch.long)
            batch_data['objdet'][indexObj]['labels'] = batch_data['objdet'][indexObj]['labels'].to(torch.long)

        imageHeight, imageWidth = batch_data['image'].shape[-2:]

        # ---------- detr net ----------
        gt_labels_forrtdetr = []
        for onegt in batch_data["gt_labels"]["objdet"]: 
            onegt = {
                "bboxes": torchvision.ops.box_convert(onegt["bboxes"][..., :4].clone(), in_fmt="xyxy", out_fmt="cxcywh"),  # Note: only left bbox is needed for detr
                "labels": onegt["labels"].clone()
            }
            onegt["bboxes"][:, [0, 2]] /= imageWidth
            onegt["bboxes"][:, [1, 3]] /= imageHeight
            gt_labels_forrtdetr.append(onegt)
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

        # @@@@@@@@@@@@@@@@@@@@ VISUALIZATION @@@@@@@@@@@@@@@@@@@@
        if tensorBoardLogger is not None:
            left_bboxes = selected_leftdetections[0]["bboxes"]
            left_bboxes = torchvision.ops.box_convert(left_bboxes.clone(), in_fmt="cxcywh", out_fmt="xyxy")
            left_bboxes[:, [0, 2]] *= imageWidth
            left_bboxes[:, [1, 3]] *= imageHeight
            leftimage_visz = RenderImageWithBboxes(
                left_event_sharp.detach().squeeze(1).cpu().numpy(),
                {
                    "bboxes": left_bboxes,
                    "classes": selected_leftdetections[0]["classes"],
                }
            )
            tensorBoardLogger.add_image("(valid) left sharp with bboxes", leftimage_visz[0])

        batchSize = batch_data["image"].shape[0]
        loss = 0
        for key, value in lossDictAll.items():
            loss += value
            if key in log_dict:
                log_dict[key].update(lossDictAll[key].item(), batchSize)
        log_dict["BestIndex"].update(loss.item(), batchSize)
        log_dict["Loss"].update(loss.item(), batchSize)

        if tensorBoardLogger is not None:
            pbar.update(1)

    torch.cuda.synchronize()

    if tensorBoardLogger is not None:
        pbar.close()

    return log_dict