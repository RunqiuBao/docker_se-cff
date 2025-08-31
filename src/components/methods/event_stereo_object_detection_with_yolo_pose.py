import os.path
import numpy
import torch
import cv2
import time
import torchvision
from typing import Optional
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torchvision.ops import nms
import torch.nn.functional as F

from tqdm import tqdm

from .visz_utils import DrawResultBboxesAndKeyptsOnStereoEventFrame, RenderImageWithBboxes
from ..models.utils.misc import freeze_module_grads, DetachCopyNested

from ..models.utils.objdet_utils import EvaluateObjDetPerformance
from ..methods.visz_utils import RenderImageWithBboxesAndKeypts
from .log_utils import GetLogDict
from .base import batch_to_cuda
from ..models.yolo_pose_utils import non_max_suppression

from..models.utils.misc import freeze_module_grads
from utils.metrics import AverageMeter

import logging
logger = logging.getLogger(__name__)


def InvertBijection(forw_mapping: Tensor) -> Tensor:
    """
    Args:
        forw_mapping: (N,), e.g. [3, 0, 1, 2]
    
    Returns:
        inv_mapping: (N,), e.g. [1, 2, 3, 0]
    """
    print("forw_mapping: {}".format(forw_mapping))
    inv_mapping = torch.empty_like(forw_mapping)
    inv_mapping[forw_mapping] = torch.arange(inv_mapping.shape[0], device=inv_mapping.device)
    return inv_mapping


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
    device: str,
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

    if loss.grad_fn is None:
        logger.error("no valid loss. skipping backward. lossDictCurrentStep:  {}".format(lossDictCurrentStep))
        return

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
        for key, model in models.items():
            if not model.module.is_freeze:
                total_norm = clip_grad_norm_(model.module.parameters(), max_norm=float('inf')) # for logging, no clip
                lossRecords["grad_norm_" + key].update(total_norm, batchSize)
        for key, suboptimizer in optimizer.items():
            if not models[key].module.is_freeze:
                suboptimizer.step()
    logger.debug("-> backward_and_optimize time cost: {}".format(time.time() - starttime))
    return


def preprocess_batch(
    batch_labels: dict,
    batch_img_metas: dict,
    is_stereo_bbox: bool,
    max_num_keypoints: int = 1
):
    """
    Prepare the batch_labels as the YoloPose required.
    """
    if is_stereo_bbox:
        bboxes_tensor = [one_labels["bboxes"][:, :] for one_labels in batch_labels]
        bboxes_tensor = torch.cat(bboxes_tensor, dim=0)
        disparity = (bboxes_tensor[:, 0] + bboxes_tensor[:, 2]) / 2 - (bboxes_tensor[:, 4] + bboxes_tensor[:, 5]) / 2
    else:
        bboxes_tensor = [one_labels["bboxes"][:, :4] for one_labels in batch_labels]
        bboxes_tensor = torch.cat(bboxes_tensor, dim=0)
        bboxes_tensor[:, [0, 2]] /= batch_img_metas["w"]
        bboxes_tensor[:, [1, 3]] /= batch_img_metas["h"]
        bboxes_tensor = torchvision.ops.box_convert(bboxes_tensor, in_fmt="xyxy", out_fmt="cxcywh")  # Note: xywh, the xy is the left top corner!!!

    cls_tensor = [one_labels["labels"] for one_labels in batch_labels]
    cls_tensor = torch.cat(cls_tensor, dim=0)
    keypts_tensor = [one_labels["keypts"] for one_labels in batch_labels]
    keypts_tensor = torch.cat(keypts_tensor, dim=0)
    if is_stereo_bbox:
        keypoints_right = keypts_tensor.clone()
        keypoints_right[:, :, 0] = keypts_tensor[:, :, 0] - disparity.unsqueeze(-1).expand(-1, max_num_keypoints)
        # Note: do not normalize for keypoints_right
    keypts_tensor[:, :, 0] /= batch_img_metas["w"]  # Note: yolo_pose format requires normalized keypoints.
    keypts_tensor[:, :, 1] /= batch_img_metas["h"]

    batchidx_tensor = [torch.ones((one_labels["bboxes"].shape[0]), dtype=torch.float, device=one_labels["bboxes"].device) * indexInBatch for indexInBatch, one_labels in enumerate(batch_labels)]
    batchidx_tensor = torch.cat(batchidx_tensor, dim=0)
    batch_labels_yolopose = {
        "bboxes": bboxes_tensor,
        "cls":  cls_tensor,
        "keypoints": keypts_tensor,
        "batch_idx": batchidx_tensor
    }
    if is_stereo_bbox:
        batch_labels_yolopose["keypoints_right"] = keypoints_right  # Note: do not normalize keypoints_right.

    return batch_labels_yolopose

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
        if model.module.is_freeze:
            model.eval()
        else:
            model.train()

    log_dict = GetLogDict(is_train=True, is_secff=(hasattr(models["disp_head"].module, 'is_freeze') and not models["disp_head"].module.is_freeze))
    # create grad norm logger
    for key, model in models.items():
        if not model.module.is_freeze:
            log_dict["grad_norm_" + key] = AverageMeter(string_format="%6.3lf")
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
        
        deviceThisProcess = batch_data["event"]["left"].device

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
        batch_img_metas = {
            "h": imageHeight,
            "w": imageWidth,
            'h_recti': batch_data['image_metadata']['h_recti'],
            'w_recti': batch_data['image_metadata']['w_recti']
        }

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
            logger.debug("data timestamp: {}, {}".format(batch_data["event"]["timestamp"][0], batch_data["objdet"][0]["timestamp"]))
            disp_map = pred_disparity_pyramid[-1].detach().cpu()
            disp_map *= 255 / disp_map.max()
            tensorBoardLogger.add_image("disp_map", disp_map[0, ...].to(torch.uint8).squeeze())
            viz_left_sharp = left_event_sharp[0].detach().squeeze().cpu()
            viz_left_sharp -= viz_left_sharp.min()
            viz_left_sharp /= viz_left_sharp.max()
            viz_left_sharp *= 255
            viz_left_sharp = RenderImageWithBboxes(
                viz_left_sharp.squeeze().cpu().numpy(),
                {
                    "bboxes": batch_data["gt_labels"]["objdet"][0]["bboxes"].detach().cpu(),
                    "classes": batch_data["gt_labels"]["objdet"][0]["labels"].detach().cpu(),
                    "confidences": torch.ones_like(batch_data["gt_labels"]["objdet"][0]["labels"]).detach().cpu()
                }
            )[0]
            tensorBoardLogger.add_image("left_sharp", viz_left_sharp.to(torch.uint8).squeeze())
            viz_right_sharp = right_event_sharp[0].detach().squeeze().cpu()
            viz_right_sharp -= viz_right_sharp.min()
            viz_right_sharp /= viz_right_sharp.max()
            viz_right_sharp *= 255
            gt_bboxes_right =  batch_data["gt_labels"]["objdet"][0]["bboxes"].detach().cpu()[:, [4, 1, 5, 3]]
            viz_right_sharp = RenderImageWithBboxes(
                viz_right_sharp.squeeze().cpu().numpy(),
                {
                    "bboxes": gt_bboxes_right,
                    "classes": batch_data["gt_labels"]["objdet"][0]["labels"].detach().cpu(),
                    "confidences": torch.ones_like(batch_data["gt_labels"]["objdet"][0]["labels"]).detach().cpu()
                }
            )[0]
            tensorBoardLogger.add_image("right_sharp", viz_right_sharp.to(torch.uint8).squeeze())
            tensorBoardLogger.add_image("disparity gt",  batch_data["gt_labels"]["disparity"].to(torch.uint8).detach().cpu()[0, ...])

        if models["disp_head"].module.is_freeze:
            objdet_targets = preprocess_batch(batch_data["gt_labels"]["objdet"], batch_img_metas, False)
            # ---------- objdet net ----------
            if not models["objdet_head"].module.is_loss_ready:
                models["objdet_head"].module.init_criterion()
            left_detections, lossDictAll, artifacts = _forward_one_batch(
                models["objdet_head"],
                {
                    "left_event_voxel": batch_data["event"]["left"],
                },
                objdet_targets,
                lossDictAll,
                {},
            )

            (
                left_selected_boxes,
                left_selected_classes,
                left_selected_confidences,
                left_selected_keypts,
                left_selected_batchidx,
                left_fg_mask,  # Note: always return even when submodel is_freeze
                left_target_gt_idx
            ) = artifacts

            # @@@@@@@@@@@@@@@@@@@@ VISUALIZATION @@@@@@@@@@@@@@@@@@@@
            if tensorBoardLogger is not None and left_selected_boxes is not None:
                leftimage_visz = RenderImageWithBboxesAndKeypts(
                    left_event_sharp[0].detach().squeeze().cpu().numpy(),
                    {
                        "bboxes": left_selected_boxes[left_selected_batchidx == 0].detach().cpu().numpy(),
                        "classes": left_selected_classes[left_selected_batchidx == 0].detach().cpu().numpy(),
                        "confidences": left_selected_confidences[left_selected_batchidx == 0].detach().cpu().numpy(),
                        "keypts": left_selected_keypts[left_selected_batchidx == 0][:, :, :].detach().cpu().numpy(),
                    }
                )
                tensorBoardLogger.add_image("(train) left sharp with bboxes", leftimage_visz)
                leftimage_gt_visz = RenderImageWithBboxesAndKeypts(
                    left_event_sharp[0].detach().squeeze().cpu().numpy(),
                    {
                        "bboxes": batch_data["gt_labels"]["objdet"][0]["bboxes"].detach().cpu().numpy(),
                        "classes": batch_data["gt_labels"]["objdet"][0]["labels"].detach().cpu().numpy(),
                        "confidences": torch.ones_like(batch_data["gt_labels"]["objdet"][0]["labels"]).cpu().numpy(),
                        "keypts": batch_data["gt_labels"]["objdet"][0]["keypts"][:, :, :2].detach().cpu().numpy(),
                    }
                )
                tensorBoardLogger.add_image("(train) left sharp with GT bboxes", leftimage_gt_visz)

            if models["objdet_head"].module.is_freeze:
                left_detections_multilevels_detachcopy = DetachCopyNested(left_detections)
                left_bboxesClsKeypts_nmsed_topked, nms_topk_mask = non_max_suppression(
                    left_detections_multilevels_detachcopy,
                    conf_thres=0.1,
                    iou_thres=0.7,
                    labels=[],
                    nc=models["objdet_head"].module.config["num_classes"],
                    multi_label=True,
                    agnostic=False,
                    max_det=models["objdet_head"].module.config["num_topk_candidates"],
                    end2end=False,
                )
                num_pos = left_fg_mask[nms_topk_mask].sum()

                if num_pos > 0:
                    # ---------- stereo detection head ----------
                    left_bboxes_nmsed_topked = [one_batch[..., :4] for one_batch in left_bboxesClsKeypts_nmsed_topked]
                    stereo_objdet_targets = preprocess_batch(
                        batch_data["gt_labels"]["objdet"],
                        batch_img_metas,
                        True,
                        1
                    )
                    stereo_preds, lossDictAll, artifacts = _forward_one_batch(
                        models["stereo_detection_head"],
                        {
                            "right_event_voxel": batch_data["event"]["right"],
                            "left_bboxes": left_bboxes_nmsed_topked,
                            "disp_prior": pred_disparity_pyramid[-1],
                            "batch_img_metas": batch_img_metas,
                            "detector_format": "yolopose"
                        },
                        {
                            "left_fg_mask": left_fg_mask,
                            "left_target_gt_idx": left_target_gt_idx,  # Note: gt index for each anchor.
                            "left_nms_topk_mask": nms_topk_mask,
                            "stereo_objdet_targets": stereo_objdet_targets,
                            "batch_img_metas": batch_img_metas
                        },
                        lossDictAll,
                        {}
                    )

                    # @@@@@@@@@@@@@@@@@@@@ VISUALIZATION @@@@@@@@@@@@@@@@@@@@
                    if tensorBoardLogger is not None:
                        if artifacts is not None:
                            for indexInBatch in range(len(artifacts[0])):
                                if artifacts[0][indexInBatch] is None:
                                    # not a valid detection
                                    continue
                                right_bboxes_one = artifacts[0][indexInBatch].detach()
                                right_bboxes_one = right_bboxes_one[..., [4, 5, 6, 7]]
                                rightimage_visz = RenderImageWithBboxesAndKeypts(
                                    right_event_sharp[indexInBatch].detach().squeeze().cpu().numpy(),
                                    {
                                        "bboxes": right_bboxes_one.detach().cpu().numpy(),
                                        "classes": -1 * torch.ones_like(right_bboxes_one[:, 0]).cpu().numpy(),
                                        "confidences": 1.0 * torch.ones_like(right_bboxes_one[:, 0]).cpu().numpy(),
                                        "keypts": artifacts[-1][indexInBatch][:, :, :2].detach().cpu().numpy() if artifacts[-1][indexInBatch] is not None else None,
                                    }
                                )
                                
                                tensorBoardLogger.add_image("(train) right sharp preds with keypts", rightimage_visz)
                                
                                right_bboxes = batch_data["gt_labels"]["objdet"][indexInBatch]["bboxes"].detach().cpu().numpy()
                                right_bboxes[:, [0, 2]] = right_bboxes[:, [4, 5]]
                                rightimage_gt_visz = RenderImageWithBboxesAndKeypts(
                                    right_event_sharp[indexInBatch].detach().squeeze().cpu().numpy(),
                                    {
                                        "bboxes": right_bboxes,
                                        "classes": batch_data["gt_labels"]["objdet"][indexInBatch]["labels"].detach().cpu().numpy(),
                                        "confidences": torch.ones_like(batch_data["gt_labels"]["objdet"][indexInBatch]["labels"]).cpu().numpy(),
                                        "keypts": batch_data["gt_labels"]["objdet"][indexInBatch]["keypts_right"][:, :, :2].detach().cpu().numpy(),
                                    }
                                )
                                tensorBoardLogger.add_image("(train) right sharp with GT bboxes", rightimage_gt_visz)
                                # # ------- debug code --------
                                # leftimage_visz = RenderImageWithBboxesAndKeypts(
                                #     left_event_sharp[indexInBatch].detach().squeeze().cpu().numpy(),
                                #     {
                                #         "bboxes": left_selected_boxes[left_selected_batchidx == indexInBatch].detach().cpu().numpy(),
                                #         "classes": left_selected_classes[left_selected_batchidx == indexInBatch].detach().cpu().numpy(),
                                #         "confidences": left_selected_confidences[left_selected_batchidx == indexInBatch].detach().cpu().numpy(),
                                #         "keypts": left_selected_keypts[left_selected_batchidx == indexInBatch][:, :, :].detach().cpu().numpy(),
                                #     }
                                # )
                                # leftimage_gt_visz = RenderImageWithBboxesAndKeypts(
                                #     left_event_sharp[indexInBatch].detach().squeeze().cpu().numpy(),
                                #     {
                                #         "bboxes": batch_data["gt_labels"]["objdet"][indexInBatch]["bboxes"].detach().cpu().numpy(),
                                #         "classes": batch_data["gt_labels"]["objdet"][indexInBatch]["labels"].detach().cpu().numpy(),
                                #         "confidences": torch.ones_like(batch_data["gt_labels"]["objdet"][indexInBatch]["labels"]).cpu().numpy(),
                                #         "keypts": batch_data["gt_labels"]["objdet"][indexInBatch]["keypts"][:, :, :2].detach().cpu().numpy(),
                                #     }
                                # )
                                # debug_path = "/root/data/debug_train/"
                                # stereo_visz = numpy.hstack([leftimage_gt_visz[:batch_data['image_metadata']['h_recti'], :batch_data['image_metadata']['w_recti']], rightimage_gt_visz[:batch_data['image_metadata']['h_recti'], :batch_data['image_metadata']['w_recti']]])
                                # cv2.imwrite(debug_path + str(indexInBatch) + "_" + str(batch_data['end_timestamp'][indexInBatch]) + ".png", stereo_visz)
                                # # ------- debug code --------

        # backward and optimize
        batchSize = batch_data["event"]["left"].shape[0]
        try:
            _backward_and_optimize(
                models,
                lossDictAll,
                optimizer,  # dict containing sub optimzers
                log_dict,
                batchSize,
                deviceThisProcess
            )
            lossDictAll = {}
        except Exception as e:
            print("Note: one image in the batch might have no valid detection.")
            # import IPython; import inspect; print('baodebug: file ({}) -- func ({})'.format(__file__, inspect.stack()[0].function)); IPython.embed()

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

    preds_evaluation = []
    targets_evaluation = []
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
        batch_img_metas = {
            "h": imageHeight,
            "w": imageWidth,
            'h_recti': batch_data['image_metadata']['h_recti'],
            'w_recti': batch_data['image_metadata']['w_recti']
        }

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
            logger.debug("data timestamp: {}, {}".format(batch_data["event"]["timestamp"][0], batch_data["objdet"][0]["timestamp"]))
            disp_map = pred_disparity_pyramid[-1].detach().cpu()
            disp_map *= 255 / disp_map.max()
            tensorBoardLogger.add_image("disp_map (valid)", disp_map[0, ...].to(torch.uint8).squeeze())
            viz_left_sharp = left_event_sharp[0].detach().squeeze().cpu()
            viz_left_sharp -= viz_left_sharp.min()
            viz_left_sharp /= viz_left_sharp.max()
            viz_left_sharp *= 255
            viz_left_sharp = RenderImageWithBboxes(
                viz_left_sharp.squeeze().cpu().numpy(),
                {
                    "bboxes": batch_data["gt_labels"]["objdet"][0]["bboxes"].detach().cpu(),
                    "classes": batch_data["gt_labels"]["objdet"][0]["labels"].detach().cpu(),
                    "confidences": torch.ones_like(batch_data["gt_labels"]["objdet"][0]["labels"]).detach().cpu()
                }
            )[0]
            tensorBoardLogger.add_image("left_sharp (valid)", viz_left_sharp.to(torch.uint8).squeeze())
            viz_right_sharp = right_event_sharp[0].detach().squeeze().cpu()
            viz_right_sharp -= viz_right_sharp.min()
            viz_right_sharp /= viz_right_sharp.max()
            viz_right_sharp *= 255
            gt_bboxes_right =  batch_data["gt_labels"]["objdet"][0]["bboxes"].detach().cpu()[:, [4, 1, 5, 3]]
            viz_right_sharp = RenderImageWithBboxes(
                viz_right_sharp.squeeze().cpu().numpy(),
                {
                    "bboxes": gt_bboxes_right,
                    "classes": batch_data["gt_labels"]["objdet"][0]["labels"].detach().cpu(),
                    "confidences": torch.ones_like(batch_data["gt_labels"]["objdet"][0]["labels"]).detach().cpu()
                }
            )[0]
            tensorBoardLogger.add_image("right_sharp (valid)", viz_right_sharp.to(torch.uint8).squeeze())
            tensorBoardLogger.add_image("disparity gt (valid)",  batch_data["gt_labels"]["disparity"].to(torch.uint8).detach().cpu()[0, ...])

        if models["disp_head"].is_freeze:
            objdet_targets = preprocess_batch(batch_data["gt_labels"]["objdet"], batch_img_metas, False)
            # ---------- objdet net ----------
            if not models["objdet_head"].is_loss_ready:
                models["objdet_head"].init_criterion()
            left_detections, lossDictAll, artifacts = _forward_one_batch(
                models["objdet_head"],
                {
                    "left_event_voxel": batch_data["event"]["left"],
                },
                objdet_targets,
                lossDictAll,
                {},
            )

            (
                left_selected_boxes,
                left_selected_classes,
                left_selected_confidences,
                left_selected_keypts,
                left_selected_batchidx,
                left_fg_mask,  # Note: always return even when submodel is_freeze
                left_target_gt_idx
            ) = artifacts

            bboxes_targets_xyxy_full = torchvision.ops.box_convert(objdet_targets["bboxes"], in_fmt="cxcywh", out_fmt="xyxy")
            bboxes_targets_xyxy_full[:, [0, 2]] *= imageWidth
            bboxes_targets_xyxy_full[:, [1, 3]] *= imageHeight
            CollectPredsTargetsForEvaluation(
                left_selected_boxes,
                left_selected_classes,
                left_selected_confidences.sigmoid(),
                left_target_gt_idx[left_fg_mask],
                bboxes_targets_xyxy_full,
                objdet_targets["cls"],
                preds_evaluation,
                targets_evaluation
            )

            # @@@@@@@@@@@@@@@@@@@@ VISUALIZATION @@@@@@@@@@@@@@@@@@@@
            if tensorBoardLogger is not None and left_selected_boxes is not None:
                leftimage_visz = RenderImageWithBboxesAndKeypts(
                    left_event_sharp[0].detach().squeeze().cpu().numpy(),
                    {
                        "bboxes": left_selected_boxes[left_selected_batchidx == 0].detach().cpu().numpy(),
                        "classes": left_selected_classes[left_selected_batchidx == 0].detach().cpu().numpy(),
                        "confidences": left_selected_confidences[left_selected_batchidx == 0].detach().cpu().numpy(),
                        "keypts": left_selected_keypts[left_selected_batchidx == 0][:, :, :].detach().cpu().numpy(),
                    }
                )
                tensorBoardLogger.add_image("(valid) left sharp with bboxes", leftimage_visz)
                leftimage_gt_visz = RenderImageWithBboxesAndKeypts(
                    left_event_sharp[0].detach().squeeze().cpu().numpy(),
                    {
                        "bboxes": batch_data["gt_labels"]["objdet"][0]["bboxes"].detach().cpu().numpy(),
                        "classes": batch_data["gt_labels"]["objdet"][0]["labels"].detach().cpu().numpy(),
                        "confidences": torch.ones_like(batch_data["gt_labels"]["objdet"][0]["labels"]).cpu().numpy(),
                        "keypts": batch_data["gt_labels"]["objdet"][0]["keypts"][:, :, :2].detach().cpu().numpy(),
                    }
                )
                tensorBoardLogger.add_image("(valid) left sharp with GT bboxes", leftimage_gt_visz)

            if models["objdet_head"].is_freeze:
                left_detections_multilevels_detachcopy = DetachCopyNested(left_detections)
                left_bboxesClsKeypts_nmsed_topked, nms_topk_mask = non_max_suppression(
                    left_detections_multilevels_detachcopy,
                    conf_thres=0.1,
                    iou_thres=0.7,
                    labels=[],
                    nc=models["objdet_head"].config["num_classes"],
                    multi_label=True,
                    agnostic=False,
                    max_det=models["objdet_head"].config["num_topk_candidates"],
                    end2end=False,
                )
                num_pos = left_fg_mask[nms_topk_mask].sum()

                if num_pos > 0:
                    # ---------- stereo detection head ----------
                    left_bboxes_nmsed_topked = [one_batch[..., :4] for one_batch in left_bboxesClsKeypts_nmsed_topked]
                    stereo_objdet_targets = preprocess_batch(
                        batch_data["gt_labels"]["objdet"],
                        batch_img_metas,
                        True,
                        1
                    )
                    stereo_preds, lossDictAll, artifacts = _forward_one_batch(
                        models["stereo_detection_head"],
                        {
                            "right_event_voxel": batch_data["event"]["right"],
                            "left_bboxes": left_bboxes_nmsed_topked,
                            "disp_prior": pred_disparity_pyramid[-1],
                            "batch_img_metas": batch_img_metas,
                            "detector_format": "yolopose"
                        },
                        {
                            "left_fg_mask": left_fg_mask,
                            "left_target_gt_idx": left_target_gt_idx,
                            "left_nms_topk_mask": nms_topk_mask,
                            "stereo_objdet_targets": stereo_objdet_targets,
                            "batch_img_metas": batch_img_metas
                        },
                        lossDictAll,
                        {}
                    )

                    # @@@@@@@@@@@@@@@@@@@@ VISUALIZATION @@@@@@@@@@@@@@@@@@@@
                    if tensorBoardLogger is not None and artifacts[0][0] is not None:
                        if artifacts is not None:
                            for indexInBatch in range(len(artifacts[0])):
                                if artifacts[0][indexInBatch] is None:
                                    # not a valid detection
                                    continue
                                right_bboxes_one = artifacts[0][indexInBatch].detach()
                                right_bboxes_one = right_bboxes_one[..., [4, 5, 6, 7]] 
                                rightimage_visz = RenderImageWithBboxesAndKeypts(
                                    right_event_sharp[indexInBatch].detach().squeeze().cpu().numpy(),
                                    {
                                        "bboxes": right_bboxes_one.detach().cpu().numpy(),
                                        "classes": -1 * torch.ones_like(right_bboxes_one[:, 0]).cpu().numpy(),
                                        "confidences": 1.0 * torch.ones_like(right_bboxes_one[:, 0]).cpu().numpy(),
                                        "keypts": artifacts[-1][indexInBatch][:, :, :2].detach().cpu().numpy() if artifacts[-1][indexInBatch] is not None else None,
                                    }
                                )

                                tensorBoardLogger.add_image("(valid) right sharp preds with keypts", rightimage_visz)
                                
                                right_bboxes = batch_data["gt_labels"]["objdet"][indexInBatch]["bboxes"].detach().cpu().numpy()
                                right_bboxes[:, [0, 2]] = right_bboxes[:, [4, 5]]
                                rightimage_gt_visz = RenderImageWithBboxesAndKeypts(
                                    right_event_sharp[indexInBatch].detach().squeeze().cpu().numpy(),
                                    {
                                        "bboxes": right_bboxes,
                                        "classes": batch_data["gt_labels"]["objdet"][indexInBatch]["labels"].detach().cpu().numpy(),
                                        "confidences": torch.ones_like(batch_data["gt_labels"]["objdet"][indexInBatch]["labels"]).cpu().numpy(),
                                        "keypts": batch_data["gt_labels"]["objdet"][indexInBatch]["keypts_right"][:, :, :2].detach().cpu().numpy(),
                                    }
                                )
                                tensorBoardLogger.add_image("(valid) right sharp with GT bboxes", rightimage_gt_visz)

                                # # ------- debug code --------
                                # debug_path = "/root/data/debug_valid/"
                                # leftimage_visz = RenderImageWithBboxesAndKeypts(
                                #     left_event_sharp[indexInBatch].detach().squeeze().cpu().numpy(),
                                #     {
                                #         "bboxes": left_selected_boxes[left_selected_batchidx == indexInBatch].detach().cpu().numpy(),
                                #         "classes": left_selected_classes[left_selected_batchidx == indexInBatch].detach().cpu().numpy(),
                                #         "confidences": left_selected_confidences[left_selected_batchidx == indexInBatch].detach().cpu().numpy(),
                                #         "keypts": left_selected_keypts[left_selected_batchidx == indexInBatch][:, :, :].detach().cpu().numpy(),
                                #     }
                                # )
                                # leftimage_gt_visz = RenderImageWithBboxesAndKeypts(
                                #     left_event_sharp[indexInBatch].detach().squeeze().cpu().numpy(),
                                #     {
                                #         "bboxes": batch_data["gt_labels"]["objdet"][indexInBatch]["bboxes"].detach().cpu().numpy(),
                                #         "classes": batch_data["gt_labels"]["objdet"][indexInBatch]["labels"].detach().cpu().numpy(),
                                #         "confidences": torch.ones_like(batch_data["gt_labels"]["objdet"][indexInBatch]["labels"]).cpu().numpy(),
                                #         "keypts": batch_data["gt_labels"]["objdet"][indexInBatch]["keypts"][:, :, :2].detach().cpu().numpy(),
                                #     }
                                # )
                                # disparity_visz = numpy.vstack([
                                #     pred_disparity_pyramid[-1][indexInBatch].detach().cpu().numpy().astype('uint8')[:batch_data['image_metadata']['h_recti'], :batch_data['image_metadata']['w_recti']],
                                #     batch_data["gt_labels"]["disparity"][indexInBatch].detach().cpu().numpy().astype('uint8')[:batch_data['image_metadata']['h_recti'], :batch_data['image_metadata']['w_recti']]
                                # ])
                                # disparity_visz = cv2.cvtColor(disparity_visz, cv2.COLOR_GRAY2BGR)
                                # stereo_visz = numpy.vstack([leftimage_gt_visz[:batch_data['image_metadata']['h_recti'], :batch_data['image_metadata']['w_recti']], rightimage_gt_visz[:batch_data['image_metadata']['h_recti'], :batch_data['image_metadata']['w_recti']]])
                                # all_visz = numpy.hstack([disparity_visz, stereo_visz])
                                # cv2.imwrite(debug_path + str(indexBatch) + "_" + str(batch_data['end_timestamp'][indexInBatch]) + ".png", all_visz)
                                # # ------- debug code --------

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

    if tensorBoardLogger is not None:
        pbar.close()
    
    evalResults = EvaluateObjDetPerformance(preds_evaluation, targets_evaluation)
    logger.info("evalResults:\n{}".format(evalResults))

    return log_dict


@torch.no_grad()
def test(
    models,
    data_loader,
    sequence_name,
    save_root,
    is_save_onnx = False
):
    for model in models.values():
        model.module.eval()

    if is_save_onnx:
        logger.info(
            '''
            # how to do inference with the exported onnx model:
            # providers = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
            import onnxruntime
            providers = ["CPUExecutionProvider"]
            ort_session = onnxruntime.InferenceSession(os.path.join(save_root, "concentration_net.onnx"), providers=providers)
            ort_inputs = {"left_event": oneInputs["event"]["left"].cpu().numpy(), "right_event": oneInputs["event"]["right"].cpu().numpy()}
            ort_outs = ort_session.run(["left_event_sharp", "right_event_sharp"], ort_inputs)
            '''
        )
    
    pbar = tqdm(total=len(data_loader))
    data_iter = iter(data_loader)
    previous_preds = None
    prediction_dict = None
    for indexBatch in range(len(data_loader.dataset)):
        batch_data = batch_to_cuda(next(data_iter))
        assert batch_data["event"]["left"].shape[0] == 1, "batch size should be 1 for test mode."
        starttime = time.time()
        # ---------- concentration net ----------
        left_event_sharp = models["concentration_net"].module.predict(batch_data["event"]["left"])
        right_event_sharp = models["concentration_net"].module.predict(batch_data["event"]["right"])
        if is_save_onnx:
            torch.onnx.export(
                models['concentration_net'].module,
                (
                    batch_data["event"]["left"],
                    batch_data["event"]["right"]
                ),
                os.path.join(save_root, "concentration_net.onnx"),
                export_params=True,
                opset_version=16,
                do_constant_folding=True,
                input_names=["left_img", "right_img"],
                output_names=["left_preds", "right_preds"]
            )

        imageHeight, imageWidth = batch_data["event"]["left"].shape[-2:]
        batch_img_metas = {
            "h": imageHeight,
            "w": imageWidth,
            'h_recti': batch_data['image_metadata']['h_recti'],
            'w_recti': batch_data['image_metadata']['w_recti']
        }
        num_classes = models["objdet_head"].module.config["num_classes"]

        # ---------- disp pred net ----------
        pred_disparity_pyramid = models["disp_head"].module.predict(left_event_sharp, right_event_sharp)
        if is_save_onnx:
            torch.onnx.export(
                models['disp_head'].module,
                (
                    left_event_sharp,
                    right_event_sharp
                ),
                os.path.join(save_root, "disp_head.onnx"),
                export_params=True,
                opset_version=16,
                do_constant_folding=True,
                input_names=["left_img", "right_img"],
                output_names=["preds"]
            )

        # ---------- objdet net ----------
        left_detections = models["objdet_head"].module.predict(batch_data["event"]["left"])
        if is_save_onnx:
            torch.onnx.export(
                models['objdet_head'].module,
                (
                    batch_data["event"]["left"],
                ),
                os.path.join(save_root, "objdet_head.onnx"),
                export_params=True,
                opset_version=16,
                do_constant_folding=True,
                input_names=["left_event_voxel", "right_event_voxel"],
                output_names=["preds0", "preds100", "preds101", "preds102", "preds11", "artifacts00", "artifacts01", "artifacts02"]
            )

        left_detections_multilevels_detachcopy = DetachCopyNested(left_detections)
        left_bboxesClsKeypts_nmsed_topked, nms_topk_mask = non_max_suppression(
            left_detections_multilevels_detachcopy,
            conf_thres=models["objdet_head"].module.config["confidence_threshold_inference"],
            iou_thres=models["objdet_head"].module.config["nms_iou_threshold_inference"],
            labels=[],
            nc=models["objdet_head"].module.config["num_classes"],
            multi_label=False,
            agnostic=False,
            max_det=models["objdet_head"].module.config["num_topk_candidates"],
            end2end=False,
        )

        refined_sbboxes_nobkg = None
        if left_bboxesClsKeypts_nmsed_topked[0].shape[0] > 0:
            # ---------- stereo detection head ----------
            left_bboxes_nmsed_topked = [one_batch[..., :4] for one_batch in left_bboxesClsKeypts_nmsed_topked]
            (
                batch_sbboxes_priors,
                batch_corresponding_leftdet_ids,
                batch_refined_right_bboxes,
                batch_refined_right_scores,
                batch_predicted_right_keypts,
                rpn_cls_scores,
                rpn_bbox_preds
            ) = models["stereo_detection_head"].module.predict(
                batch_data["event"]["right"],
                left_bboxes_nmsed_topked,
                pred_disparity_pyramid[-1],
                batch_img_metas
            )
            if is_save_onnx:
                torch.onnx.export(
                    models['stereo_detection_head'].module,
                    (
                        batch_data["event"]["right"],
                        left_bboxes_nmsed_topked,
                        pred_disparity_pyramid[-1],
                        batch_img_metas,
                    ),
                    os.path.join(save_root, "stereo_detection_head.onnx"),
                    export_params=True,
                    opset_version=16,
                    do_constant_folding=True,
                    input_names=["right_feat", "left_bboxes", "disp_prior", "batch_img_metas"],
                    output_names=["batch_sbboxes_priors", "batch_refined_right_bboxes", "batch_refined_right_scores", "batch_predicted_right_keypts"]
                )

            assert left_event_sharp.shape[0] == 1  # batch size should be 1
            if batch_sbboxes_priors[0] is not None:
                # select best right bboxes and keypts for visualization.
                (
                    mask_nonbackground,
                    refined_sbboxes_nobkg,
                    refined_right_scored_pred,
                    right_keypts_pred_nobkg
                ) = models["stereo_detection_head"].module.extract_inference_results(
                    batch_sbboxes_priors[0].squeeze(0),
                    batch_refined_right_bboxes[0].view(-1, num_classes, 4),
                    batch_refined_right_scores[0],
                    batch_predicted_right_keypts[0].view(-1, num_classes, models["stereo_detection_head"].module.config["max_num_keypoints"] * 3)
                )

        logger.info("one infer time: {} sec.".format(time.time() - starttime))
        if is_save_onnx and (left_bboxesClsKeypts_nmsed_topked[0].shape[0] > 0):
            print("==================================== finished onnx model (event_stereo_object_detection_with_yolo_pose) export! ====================================")
            break

        if refined_sbboxes_nobkg is not None:
            # (l_tl_x, l_tl_y, l_br_x, l_br_y,
            #                                  r_tl_x, r_tl_y, r_br_x, r_br_y,
            #                                                                 class_label, confidence, confidence_right,
            #                                                                                                           l_kpt0_x, l_kpt0_y, visibility_l0, l_kpt1_x, l_kpt1_y, visibility_l1, ..., r_kpt0_x, r_kpt0_y, visibility_r0, r_kpt1_x, r_kpt1_y, visibility_r1, ...)
            # TODO: mark right confidences on the result visz image.
            corresponding_leftdet_indices = batch_corresponding_leftdet_ids[0][mask_nonbackground]
            corresponding_leftdets = left_bboxesClsKeypts_nmsed_topked[0][corresponding_leftdet_indices]
            left_bboxes_final = corresponding_leftdets[:, :4]
            # limit keypts y to bbox range.
            right_keypts_pred_nobkg = right_keypts_pred_nobkg.view(-1, models["stereo_detection_head"].module.config["max_num_keypoints"] * 3)
            right_keypts_pred_nobkg[:, 1::3] = torch.clamp(
                right_keypts_pred_nobkg[:, 1::3],
                min=left_bboxes_final[:, 1].unsqueeze(-1),
                max=left_bboxes_final[:, 3].unsqueeze(-1)
            )
            left_keypts_pred = corresponding_leftdets[:, (4 + models["objdet_head"].module.config["num_classes"]):]
            left_keypts_pred[:, 1::3] = torch.clamp(
                left_keypts_pred[:, 1::3],
                min=left_bboxes_final[:, 1].unsqueeze(-1),
                max=left_bboxes_final[:, 3].unsqueeze(-1)
            )
            raw_preds = torch.concat([
                left_bboxes_final,
                torch.concat([
                    refined_sbboxes_nobkg[:, 4].view(-1, 1),
                    left_bboxes_final[:, 1].view(-1, 1),
                    refined_sbboxes_nobkg[:, 6].view(-1, 1),
                    left_bboxes_final[:, 3].view(-1, 1)
                ], dim=-1),
                torch.argmax(corresponding_leftdets[:, 4:(4 + num_classes)], dim=-1).unsqueeze(-1),
                torch.max(corresponding_leftdets[:, 4:(4 + num_classes)], dim=-1)[0].unsqueeze(-1),
                refined_right_scored_pred.view(-1, 1),
                left_keypts_pred,
                right_keypts_pred_nobkg,
            ], dim=1)
            right_keep_indices = nms(
                raw_preds[:, 4:8],
                raw_preds[:, 10],
                iou_threshold=models["stereo_detection_head"].module.config["right_nms_iou_threshold_inference"]
            )
            raw_preds = raw_preds[right_keep_indices]
            left_keep_indices = nms(
                raw_preds[:, 0:4],
                raw_preds[:, 9],
                iou_threshold=models["objdet_head"].module.config["confidence_threshold_inference"]
            )
            raw_preds = raw_preds[left_keep_indices]

            # # -------------- debug code --------------
            # dummy_results = left_bboxesClsKeypts_nmsed_topked[0][:, :4]
            # warped_left_bboxes = models["stereo_detection_head"].module.warp_bboxes(
            #     [dummy_results],
            #     pred_disparity_pyramid[-1],
            #     imageHeight,
            #     imageWidth
            # )[0]
            # dummy_results = torch.concat([
            #     dummy_results[:, :4],
            #     warped_left_bboxes
            # ], dim=-1)
            # dummy_results = torch.concat([
            #     dummy_results,
            #     torch.ones((dummy_results.shape[0], 3), device=dummy_results.device, dtype=dummy_results.dtype),
            #     torch.zeros((dummy_results.shape[0], 12), device=dummy_results.device, dtype=dummy_results.dtype),
            # ], dim=-1)
            # preds = dummy_results
            # # -------------- debug code --------------

            preds = FilterBadDetections(
                raw_preds,
                imageHeight=batch_data["image_metadata"]["h_recti"],
                imageWidth=batch_data["image_metadata"]["w_recti"],
                margin=4,
                right_confidence_threshold=models["stereo_detection_head"].module.config["right_confidence_threshold_inference"],
                left_right_confidence_diff=models["stereo_detection_head"].module.config["left_right_confidence_diff_inference"],
                left_right_width_diff_threshold=models["stereo_detection_head"].module.config["left_right_width_diff_threshold"],
            )

            if preds is not None:
                prediction_dict = {
                    "objdet": [preds],
                    "concentrate": {
                        "left": left_event_sharp,
                        "right": right_event_sharp
                    },
                    "ts": batch_data["end_timestamp"].item(),
                    "disp": cv2.cvtColor(pred_disparity_pyramid[-1].detach().cpu().numpy().astype('uint8')[0], cv2.COLOR_GRAY2BGR)
                }
                stereo_visz = SaveTestResultsAndVisualize(
                    prediction_dict,
                    indexBatch,
                    batch_data["end_timestamp"].item(),
                    sequence_name,
                    save_root,
                    batch_data["image_metadata"]
                )
                # # -------------- debug code --------------
                # os.makedirs("/root/data/debug_test/", exist_ok=True)
                # h, w = stereo_visz[0].shape[:2]
                # h = h // 2
                # cv2.imwrite("/root/data/debug_test/" + str(prediction_dict['ts']) + ".png", numpy.vstack([prediction_dict['disp'][:h, :w], stereo_visz[0]]))
                # # -------------- debug code --------------

                previous_preds = preds
            else:
                logger.error("batch {} has no valid detections.".format(indexBatch))
        else:
            logger.error("batch {} has no valid detections.".format(indexBatch))

        pbar.update(1)
    pbar.close()
    return


def FilterIrregularBboxes(preds: Tensor, hw_ratiorange_class0: list):
    if preds is None:
        return None
    new_preds = []
    for indexPred in range(preds.shape[0]):
        pred = preds[indexPred]
        print(
            (pred[3] - pred[1]) / (pred[2] - pred[0])
        )
        hw_ratio = (pred[3] - pred[1]) / (pred[2] - pred[0])
        if hw_ratio < hw_ratiorange_class0[0] or hw_ratio > hw_ratiorange_class0[1]:
            print("bbox {} is filtered out due to irregular hw ratio: {}".format(pred, hw_ratio))
            continue
        new_preds.append(pred.unsqueeze(0))
    return torch.concat(new_preds, dim=0) if len(new_preds) > 0 else None


def FilterBadDetections(preds: Tensor, imageHeight: int, imageWidth: int, margin: int, right_confidence_threshold: float, left_right_confidence_diff: float, left_right_width_diff_threshold: float):
    """
    delete objects whose bboxes are within 4 edges' margin of the image.
    delete objects whose keypoints are outside of the bbox.
    """
    new_preds = []
    num_objects = preds.shape[0]
    max_num_keypoints = (preds.shape[1] - 10) // 3 // 2
    for i in range(num_objects):
        if (
            preds[i][0] < margin
            or preds[i][1] < margin
            or preds[i][0] > (imageWidth - margin)
            or preds[i][1] > (imageWidth - margin)
            or preds[i][4] < margin
            or preds[i][5] < margin
            or preds[i][4] > (imageWidth - margin)
            or preds[i][5] > (imageWidth - margin)
            or preds[i][2] < margin
            or preds[i][3] < margin
            or preds[i][2] > (imageWidth - margin)
            or preds[i][3] > (imageHeight - margin)
            or preds[i][6] < margin
            or preds[i][7] < margin
            or preds[i][6] > (imageWidth - margin)
            or preds[i][7] > (imageWidth - margin)
        ):
            print("{}-th object is filtered out due to inside image edge margin.".format(i))
            continue
        if preds[i][10] < right_confidence_threshold:
            print("{}-th object is filtered out due to low right confidence.".format(i))
            continue
        if abs(preds[i][9] - preds[i][10]) > left_right_confidence_diff:
            print("{}-th object is filtered out due to too much left right confidence diff.".format(i))
            continue
        if (preds[i][2] - preds[i][0]) * (preds[i][3] - preds[i][1]) < 600:  # bbox area should be larger than 2000 pixels
            print("{}-th object is filtered out due to too small bbox area.".format(i))
            continue
        # filter width change:
        left_width = preds[i][2] - preds[i][0]
        right_width = preds[i][6] - preds[i][4]
        if (max(left_width, right_width) / min(left_width, right_width)) > left_right_width_diff_threshold:
            print("{}-th object is filtered out due to too much left right width diff.".format(i))
            continue

        # isKeyptsOutsideBbox = False
        # for indexKeypt in range(max_num_keypoints):
        #     if preds[i][11 + indexKeypt * 3 + 2] > 0:
        #         # keypoint is visible
        #         if (
        #             preds[i][11 + indexKeypt * 3] < preds[i][0]
        #             or preds[i][11 + indexKeypt * 3] > preds[i][2]
        #             or preds[i][11 + indexKeypt * 3 + 1] < preds[i][1]
        #             or preds[i][11 + indexKeypt * 3 + 1] > preds[i][3]
        #             or preds[i][11 + max_num_keypoints * 3 + indexKeypt * 3] < preds[i][4]
        #             or preds[i][11 + max_num_keypoints * 3 + indexKeypt * 3] > preds[i][6]
        #             or preds[i][11 + max_num_keypoints * 3 + indexKeypt * 3 + 1] < preds[i][5]
        #             or preds[i][11 + max_num_keypoints * 3 + indexKeypt * 3 + 1] > preds[i][7]
        #         ):
        #             isKeyptsOutsideBbox = True
        #             break
        # if isKeyptsOutsideBbox:
        #     print("{}-th object is filtered out due to keypoints outside bbox.".format(i))
        #     continue
        new_preds.append(preds[i].unsqueeze(0))
        print("{}-th object is kept.".format(i))
    if len(new_preds) > 0:
        return torch.concat(new_preds, dim=0)
    else:
        return None
    

def FilterTemporal(preds: Tensor, ref_preds: Tensor, iou_threshold_for_matching: float, iou_leftright_for_filtering: float, area_change_threshold: float):
    """
    temporal filtering based on area change of the bboxes. Filter out the suddenly-small bboxes.
    """
    if ref_preds is None or preds is None:
        return preds
    
    new_preds = []
    for pred in preds:
        found_match = False
        for ref_pred in ref_preds:
            max_x_tl = max(pred[0], ref_pred[0])
            max_y_tl = max(pred[1], ref_pred[1])
            min_x_br = min(pred[2], ref_pred[2])
            min_y_br = min(pred[3], ref_pred[3])
            overlap_area = (min_x_br - max_x_tl) * (min_y_br - max_y_tl)
            union_area = (pred[2] - pred[0]) * (pred[3] - pred[1]) + (ref_pred[2] - ref_pred[0]) * (ref_pred[3] - ref_pred[1]) - overlap_area
            iou = max(overlap_area / union_area, 0)
            if iou < iou_threshold_for_matching:
                continue
            # found match
            found_match = True
            area_pred = (pred[2] - pred[0]) * (pred[3] - pred[1])
            area_previous = (ref_pred[2] - ref_pred[0]) * (ref_pred[3] - ref_pred[1])
            max_x_tl_r = max(pred[4], ref_pred[4])
            max_y_tl_r = max(pred[5], ref_pred[5])
            min_x_br_r = min(pred[6], ref_pred[6])
            min_y_br_r = min(pred[7], ref_pred[7])
            overlap_area_r = (min_x_br_r - max_x_tl_r) * (min_y_br_r - max_y_tl_r)
            union_area = (pred[6] - pred[4]) * (pred[7] - pred[5]) + (ref_pred[6] - ref_pred[4]) * (ref_pred[7] - ref_pred[5]) - overlap_area_r
            iou_right = max(overlap_area_r / union_area, 0)
            if area_pred < area_previous * area_change_threshold:
                break
            elif iou > iou_leftright_for_filtering and iou_right < iou_leftright_for_filtering:
                break
            else:
                new_preds.append(pred.unsqueeze(0))
                break
        if not found_match:
            new_preds.append(pred.unsqueeze(0))
    logger.info("temporalFilter: found {} valid preds in {} preds".format(len(new_preds), preds.shape[0]))
    return torch.concat(new_preds, dim=0) if len(new_preds) > 0 else None


@torch.no_grad
def CollectPredsTargetsForEvaluation(
    selected_boxes,
    selected_classes,
    selected_scores,
    selected_target_gt_idx,
    gt_bboxes,
    gt_cls,
    preds_evaluation,  # Output
    targets_evaluation
):
    """
    Prepare preds and targets pair for final evaluation. See requirements at objdet_utils.EvaluateObjDetPerformance
    """
    num_gt = gt_bboxes.shape[0]
    preds_bboxes = []
    preds_scores = []
    preds_labels = []
    target_bboxes = []
    target_cls = []
    for ii in range(num_gt):
        mask_for_this_gt = selected_target_gt_idx == ii
        selected_score_for_this_gt = selected_scores[mask_for_this_gt]
        if mask_for_this_gt.sum() == 0:
            continue
        index_highest_score = torch.argmax(selected_score_for_this_gt)
        preds_bboxes.append(selected_boxes[mask_for_this_gt][index_highest_score].unsqueeze(0))
        preds_scores.append(selected_score_for_this_gt[index_highest_score].view(1))
        preds_labels.append(selected_classes[mask_for_this_gt][index_highest_score].view(1))
        target_bboxes.append(gt_bboxes[ii].unsqueeze(0))
        target_cls.append(gt_cls[ii].view(1))
    preds_evaluation.append({
        "boxes": torch.concat(preds_bboxes, dim=0),
        "scores": torch.concat(preds_scores),
        "labels": torch.concat(preds_labels)
    })
    targets_evaluation.append({
        "boxes": torch.concat(target_bboxes, dim=0),
        "labels": torch.concat(target_cls, dim=0),
    })
    return


def SaveTestResultsAndVisualize(pred: dict, indexBatch: int, timestamp: int, sequence_name: str, save_root: str, img_metas: dict):
    """
    for EventStereoObjectDetectionNetwork.
    Args:
        pred:
            objdet: List[Tensor]. Each tensor is a (NumInstance, 10 + 3*numKeypts*2) shape.
            concentrate: Dict[Tensor]. "left" and "right", each is a (B, 1, H, W) shape tensor.
        indexBatch: index of the batch in dataset.
        timestamp: timestamp of the batch.
        sequence_name: name of the data sequence. for saving.
        save_root: path.
        img_metas: "h" and "w" of each input frame. Usaually contains padding.
    """
    # save detection results in txt, frame by frame.
    path_det_results_folder = os.path.join(save_root, "inference", "detections", sequence_name)
    path_tsfile = os.path.join(save_root, "inference", "detections", sequence_name, "timestamps.txt")
    os.makedirs(path_det_results_folder, exist_ok=True)
    ts_openmode = "a" if os.path.isfile(path_tsfile) else "w"
    with open(path_tsfile, ts_openmode) as tsfile:
        tsfile.write(str(timestamp) + "\n")
    detresults_openmode = "w"
    batch_size = len(pred['objdet'])
    for indexInBatch, detection in enumerate(pred['objdet']):
        with open(os.path.join(path_det_results_folder, str(timestamp) + ".txt"), detresults_openmode) as detresult_file:
            for indexDet in range(detection.shape[0]):
                oneDet = numpy.array2string(detection[indexDet].cpu().numpy(), separator=" ", max_line_width=numpy.inf, formatter={'float_kind':lambda x: "%.4f" % x})[1:-1]
                detresult_file.write(oneDet + "\n")

    # save detection visualization results, frame by frame. Concate left and right horizontally.
    path_det_visz_folder = os.path.join(save_root, "inference", "det_visz", sequence_name)
    path_concentrate_left_folder = os.path.join(save_root, "inference", "left", sequence_name)
    path_concentrate_right_folder = os.path.join(save_root, "inference", "right", sequence_name)
    os.makedirs(path_det_visz_folder, exist_ok=True)
    os.makedirs(path_concentrate_left_folder, exist_ok=True)
    os.makedirs(path_concentrate_right_folder, exist_ok=True)
    imgHeight, imgWidth = img_metas['h_recti'], img_metas['w_recti']
    facets_info_batch = []
    stereo_visz = []
    for indexInBatch, detection in enumerate(pred['objdet']):
        max_num_keypoints = (detection.shape[1] - 10) // 3 // 2
        left_bboxes = detection[:, 0:4].cpu().numpy()
        tl_x = numpy.clip(left_bboxes[:, 0], 0, imgWidth)
        tl_y = numpy.clip(left_bboxes[:, 1], 0, imgHeight)
        br_x = numpy.clip(left_bboxes[:, 2], 0, imgWidth)
        br_y = numpy.clip(left_bboxes[:, 3], 0, imgHeight)
        left_bboxes = numpy.stack([tl_x, tl_y, br_x, br_y], axis=1)
        right_bboxes = detection[:, 4:8].cpu().numpy()
        tl_x_r = numpy.clip(right_bboxes[:, 0], 0, imgWidth)
        tl_y_r = numpy.clip(right_bboxes[:, 1], 0, imgHeight)
        br_x_r = numpy.clip(right_bboxes[:, 2], 0, imgWidth)
        br_y_r = numpy.clip(right_bboxes[:, 3], 0, imgHeight)
        right_bboxes = numpy.stack([tl_x_r, tl_y_r, br_x_r, br_y_r], axis=1)
        sbboxes = numpy.concatenate([left_bboxes, right_bboxes], axis=-1)
        classes = detection[:, 8].cpu().numpy().astype('int')
        confidences = detection[:, 9:11].cpu().numpy()
        if detection.shape[-1] > 11:
            keypts_left = detection[:, 11:(11+max_num_keypoints*3)].cpu().numpy()
            keypts_right = detection[:, (11+max_num_keypoints*3):].cpu().numpy()
            visz_left, visz_right = DrawResultBboxesAndKeyptsOnStereoEventFrame(
                pred['concentrate']['left'].squeeze().cpu().numpy()[:img_metas['h_recti'], :img_metas['w_recti']],
                pred['concentrate']['right'].squeeze().cpu().numpy()[:img_metas['h_recti'], :img_metas["w_recti"]],
                sbboxes,
                classes,
                confidences[:, 0],
                keypts_left,
                keypts_right,
                stereo_confidences=confidences[:, 1],
            )
        visz = numpy.concatenate([visz_left, visz_right], axis=-2)
        visz = cv2.cvtColor(visz, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(path_det_visz_folder, str(timestamp) + ".png"), visz)
        left_concentrated = pred['concentrate']['left'].squeeze().cpu().numpy()[:img_metas['h_recti'], :img_metas['w_recti']]
        left_concentrated = left_concentrated - left_concentrated.min()
        left_concentrated = (left_concentrated * 255 / left_concentrated.max()).astype('uint8')
        right_concentrated = pred['concentrate']['right'].squeeze().cpu().numpy()[:img_metas['h_recti'], :img_metas['w_recti']]
        right_concentrated = right_concentrated - right_concentrated.min()
        right_concentrated = (right_concentrated * 255 / right_concentrated.max()).astype('uint8')
        cv2.imwrite(os.path.join(path_concentrate_left_folder, str(timestamp) + ".png"), left_concentrated)
        cv2.imwrite(os.path.join(path_concentrate_right_folder, str(timestamp) + ".png"), right_concentrated)
        stereo_visz.append(numpy.vstack([visz_left, visz_right]))
    return stereo_visz
