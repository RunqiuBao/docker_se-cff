import os.path
import numpy
import torch.nn.functional as F
import torch
import torch.distributed as dist
import cv2
import time
import torchvision
from typing import Optional
from torch import Tensor

from tqdm import tqdm
from collections import OrderedDict

from utils import visualizer
from .visz_utils import DrawResultBboxesAndKeyptsOnStereoEventFrame, RenderImageWithBboxes
from ..models.utils.misc import freeze_module_grads, convert_tensor_to_numpy, DetachCopyNested

from ..models.utils.objdet_utils import SelectTargets, SelectTopkCandidates
from ..methods.visz_utils import RenderImageWithBboxesAndKeypts
from .log_utils import GetLogDict
from .base import batch_to_cuda
from ..models.yolo_pose_utils import non_max_suppression

from..models.utils.misc import freeze_module_grads

import logging
logger = logging.getLogger(__name__)


def ExtractRefinedInstance(refined_instances: Tensor, refined_scores: Tensor):
    """
    for each instance, find the highest scored one in ker_h * ker_w and select the corresponding one as the final.
    Args:
        refined_instances: (B, NumInstance, ker_h * ker_w, ?).
        refined_scores: (B, NumInstance, ker_h * ker_w, 1).
    """
    logits_length = refined_instances.shape[-1]
    indices_highest_score = torch.argmax(refined_scores.squeeze(-1), dim=-1)
    refined_instances_selected = torch.gather(refined_instances, -2, indices_highest_score.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, logits_length)).squeeze(-2)
    return refined_instances_selected


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

    if loss.grad_fn is None:
        logger.error("no valid loss. skipping backward.")
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
        for key, suboptimizer in optimizer.items():
            if not models[key].module.is_freeze:
                suboptimizer.step()
    logger.debug("-> backward_and_optimize time cost: {}".format(time.time() - starttime))
    return


def preprocess_batch(batch_labels: dict, batch_img_metas: dict, is_stereo_bbox: bool):
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
        keypoints_right[:, :, 0] = keypts_tensor[:, :, 0] - disparity.unsqueeze(-1).expand(-1, 2)
        # Note: do not normalize for keypoints_right
    keypts_tensor[:, :, 0] /= batch_img_metas["w"]
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
        batch_labels_yolopose["keypoints_right"] = keypoints_right
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
        batch_img_metas = {"h": imageHeight, "w": imageWidth}

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
            objdet_targets = preprocess_batch(batch_data["gt_labels"]["objdet"], batch_img_metas, False)
            # ---------- objdet net ----------
            if not models["objdet_head"].module.is_loss_ready:
                models["objdet_head"].module.init_criterion()
            left_detections, lossDictAll, artifacts = _forward_one_batch(
                models["objdet_head"],
                {
                    "left_event_voxel": batch_data["event"]["left"],
                    "right_event_voxel": batch_data["event"]["right"],
                },
                objdet_targets,
                lossDictAll,
                {},
            )

            (
                right_feature,
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
                        "keypts1": left_selected_keypts[left_selected_batchidx == 0][:, 0, :].detach().cpu().numpy(),
                        "keypts2": left_selected_keypts[left_selected_batchidx == 0][:, 1, :].detach().cpu().numpy()
                    }
                )
                tensorBoardLogger.add_image("(train) left sharp with bboxes", leftimage_visz)

            if models["objdet_head"].module.is_freeze:
                left_detections_multilevels_detachcopy = DetachCopyNested(left_detections)
                left_bboxesClsKeypts_nmsed_topked, nms_topk_mask = non_max_suppression(
                    left_detections_multilevels_detachcopy,
                    conf_thres=0.1,
                    iou_thres=0.7,
                    labels=[],
                    nc=1,
                    multi_label=True,
                    agnostic=False,
                    max_det=models["objdet_head"].module.config["num_topk_candidates"],
                    end2end=False,
                )
                num_pos = left_fg_mask[nms_topk_mask].sum()

                if num_pos > 0:
                    # ---------- stereo detection head ----------
                    stereo_objdet_targets = preprocess_batch(batch_data["gt_labels"]["objdet"], batch_img_metas, True)
                    left_bboxes_nmsed_topked = [one_batch[..., :4] for one_batch in left_bboxesClsKeypts_nmsed_topked]
                    stereo_preds, lossDictAll, artifacts = _forward_one_batch(
                        models["stereo_detection_head"],
                        {
                            "right_feat": right_feature,
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
                    if tensorBoardLogger is not None:
                        if artifacts is not None:
                            pos_masks_one = artifacts[1][0]
                            right_bboxes_one = artifacts[0][0].detach()
                            right_bboxes_one = torch.cat([
                                right_bboxes_one[..., 4].unsqueeze(-1),
                                right_bboxes_one[..., 1].unsqueeze(-1),
                                right_bboxes_one[..., 5].unsqueeze(-1),
                                right_bboxes_one[..., 3].unsqueeze(-1)
                            ], dim=-1)
                            rightimage_visz = RenderImageWithBboxesAndKeypts(
                                right_event_sharp[0].detach().squeeze().cpu().numpy(),
                                {
                                    "bboxes": right_bboxes_one[pos_masks_one].detach().cpu().numpy(),
                                    "classes": left_selected_classes[left_selected_batchidx == 0].detach().cpu().numpy(),
                                    "confidences": left_selected_confidences[left_selected_batchidx == 0].detach().cpu().numpy(),
                                    "keypts1": artifacts[-1][0][:, 0, :2].detach().cpu().numpy(),
                                    "keypts2": artifacts[-1][0][:, 1, :2].detach().cpu().numpy()
                                }
                            )
                            tensorBoardLogger.add_image("(train) right sharp preds with keypts", rightimage_visz)

                            left_bboxes_one = left_bboxes_nmsed_topked[0][pos_masks_one]
                            leftimage_visz = RenderImageWithBboxes(
                                left_event_sharp[0].detach().squeeze(1).cpu().numpy(),
                                {
                                    "bboxes": left_bboxes_one,
                                    "classes": left_selected_classes[left_selected_batchidx == 0],
                                }
                            )
                            tensorBoardLogger.add_image("(train) left sharp preds", leftimage_visz[0])

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
        batch_img_metas = {"h": imageHeight, "w": imageWidth}

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
            objdet_targets = preprocess_batch(batch_data["gt_labels"]["objdet"], batch_img_metas, False)
            # ---------- objdet net ----------
            if not models["objdet_head"].is_loss_ready:
                models["objdet_head"].init_criterion()
            left_detections, lossDictAll, artifacts = _forward_one_batch(
                models["objdet_head"],
                {
                    "left_event_voxel": batch_data["event"]["left"],
                    "right_event_voxel": batch_data["event"]["right"],
                },
                objdet_targets,
                lossDictAll,
                {},
            )

            (
                right_feature,
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
                        "keypts1": left_selected_keypts[left_selected_batchidx == 0][:, 0, :].detach().cpu().numpy(),
                        "keypts2": left_selected_keypts[left_selected_batchidx == 0][:, 1, :].detach().cpu().numpy()
                    }
                )
                tensorBoardLogger.add_image("(valid) left sharp with bboxes", leftimage_visz)

            if models["objdet_head"].is_freeze:
                left_detections_multilevels_detachcopy = DetachCopyNested(left_detections)
                left_bboxesClsKeypts_nmsed_topked, nms_topk_mask = non_max_suppression(
                    left_detections_multilevels_detachcopy,
                    conf_thres=0.001,
                    iou_thres=0.7,
                    labels=[],
                    nc=1,
                    multi_label=True,
                    agnostic=False,
                    max_det=models["objdet_head"].config["num_topk_candidates"],
                    end2end=False,
                )
                num_pos = left_fg_mask[nms_topk_mask].sum()

                if num_pos > 0:
                    # ---------- stereo detection head ----------
                    stereo_objdet_targets = preprocess_batch(batch_data["gt_labels"]["objdet"], batch_img_metas, True)
                    left_bboxes_nmsed_topked = [one_batch[..., :4] for one_batch in left_bboxesClsKeypts_nmsed_topked]
                    stereo_preds, lossDictAll, artifacts = _forward_one_batch(
                        models["stereo_detection_head"],
                        {
                            "right_feat": right_feature,
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
                    if tensorBoardLogger is not None:
                        if artifacts is not None:
                            pos_masks_one = artifacts[1][0]
                            right_bboxes_one = artifacts[0][0].detach()
                            right_bboxes_one = torch.cat([
                                right_bboxes_one[..., 4].unsqueeze(-1),
                                right_bboxes_one[..., 1].unsqueeze(-1),
                                right_bboxes_one[..., 5].unsqueeze(-1),
                                right_bboxes_one[..., 3].unsqueeze(-1)
                            ], dim=-1)
                            rightimage_visz = RenderImageWithBboxesAndKeypts(
                                right_event_sharp[0].detach().squeeze().cpu().numpy(),
                                {
                                    "bboxes": right_bboxes_one[pos_masks_one].detach().cpu().numpy(),
                                    "classes": left_selected_classes[left_selected_batchidx == 0].detach().cpu().numpy(),
                                    "confidences": left_selected_confidences[left_selected_batchidx == 0].detach().cpu().numpy(),
                                    "keypts1": artifacts[-1][0][:, 0, :2].detach().cpu().numpy(),
                                    "keypts2": artifacts[-1][0][:, 1, :2].detach().cpu().numpy()
                                }
                            )
                            tensorBoardLogger.add_image("(valid) right sharp preds with keypts", rightimage_visz)

                            left_bboxes_one = left_bboxes_nmsed_topked[0][pos_masks_one]
                            leftimage_visz = RenderImageWithBboxes(
                                left_event_sharp[0].detach().squeeze(1).cpu().numpy(),
                                {
                                    "bboxes": left_bboxes_one,
                                    "classes": left_selected_classes[left_selected_batchidx == 0],
                                }
                            )
                            tensorBoardLogger.add_image("(valid) left sharp preds", leftimage_visz[0])


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


@torch.no_grad()
def test(
    models,
    data_loader,
    sequence_name,
    save_root,
    is_save_onnx = False
):
    for model in models.values():
        model.eval()
    
    pbar = tqdm(total=len(data_loader))
    data_iter = iter(data_loader)
    for indexBatch in range(len(data_loader.dataset)):
        batch_data = batch_to_cuda(next(data_iter))
        starttime = time.time()
        # ---------- concentration net ----------
        left_event_sharp = models["concentration_net"].module.predict(batch_data["event"]["left"])
        right_event_sharp = models["concentration_net"].module.predict(batch_data["event"]["right"])

        imageHeight, imageWidth = batch_data["event"]["left"].shape[-2:]
        batch_img_metas = {"h": imageHeight, "w": imageWidth}
        num_classes = models["objdet_head"].module.config["num_classes"]

        # ---------- disp pred net ----------
        pred_disparity_pyramid = models["disp_head"].module.predict(left_event_sharp, right_event_sharp)

        # ---------- objdet net ----------
        left_detections = models["objdet_head"].module.predict(batch_data["event"]["left"])
        right_feature = models["objdet_head"].module.predict(batch_data["event"]["right"], isRightFeatures=True)

        left_detections_multilevels_detachcopy = DetachCopyNested(left_detections)
        left_bboxesClsKeypts_nmsed_topked, nms_topk_mask = non_max_suppression(
            left_detections_multilevels_detachcopy,
            conf_thres=0.52,
            iou_thres=0.7,
            labels=[],
            nc=1,
            multi_label=True,
            agnostic=False,
            max_det=models["objdet_head"].module.config["num_topk_candidates"],
            end2end=False,
        )

        batch_refined_right_bboxes_selected = None
        if left_bboxesClsKeypts_nmsed_topked[0].shape[0] > 0:
            # ---------- stereo detection head ----------
            left_bboxes_nmsed_topked = [one_batch[..., :4] for one_batch in left_bboxesClsKeypts_nmsed_topked]
            batch_sbboxes_pred, batch_refined_right_bboxes, batch_right_scores_refine, right_pred_kpts, right_scores_keypts = models["stereo_detection_head"].module.predict(
                right_feature,
                left_bboxes_nmsed_topked,
                pred_disparity_pyramid[-1],
                batch_img_metas,
                models["stereo_detection_head"].module.config["bbox_expand_factor"],
                True
            )

            assert left_event_sharp.shape[0] == 1  # batch size should be 1
            batch_refined_right_bboxes_selected = ExtractRefinedInstance(batch_refined_right_bboxes[0], batch_right_scores_refine[0])
            batch_refined_right_keypts_selected = ExtractRefinedInstance(right_pred_kpts[0], right_scores_keypts[0])

        logger.info("one infer time: {} sec.".format(time.time() - starttime))

        if batch_refined_right_bboxes_selected is not None:
            # (l_tl_x, l_tl_y, r_br_x, r_br_y,
            #                                  r_tl_x, r_tl_y, r_br_x, r_br_y,
            #                                                                  l_kpt0_x, l_kpt0_y, l_kpt1_x, l_kpt1_y,
            #                                                                                                          r_kpt0_x, r_kpt0_y, r_kpt1_x, r_kpt1_y, class_label, confidence)
            prediction_dict = {
                "objdet": [
                    torch.concat([
                        left_bboxesClsKeypts_nmsed_topked[0][:, 0:4],
                        batch_refined_right_bboxes_selected[0],
                        left_bboxesClsKeypts_nmsed_topked[0][:, 5:7],
                        left_bboxesClsKeypts_nmsed_topked[0][:, 8:10],
                        batch_refined_right_keypts_selected[0][:, 0:2],
                        batch_refined_right_keypts_selected[0][:, 3:5],
                        torch.argmax(left_bboxesClsKeypts_nmsed_topked[0][:, 4:(4 + num_classes)].unsqueeze(-1), dim=-1),
                        torch.max(left_bboxesClsKeypts_nmsed_topked[0][:, 4:(4 + num_classes)].unsqueeze(-1), dim=-1)[0],
                    ], dim=1)
                ],
                "concentrate": {
                    "left": left_event_sharp,
                    "right": right_event_sharp
                }
            }
            SaveTestResultsAndVisualize(
                prediction_dict,
                indexBatch,
                batch_data["end_timestamp"].item(),
                sequence_name,
                save_root,
                batch_data["image_metadata"]
            )

        pbar.update(1)
    pbar.close()
    return


def SaveTestResultsAndVisualize(pred: dict, indexBatch: int, timestamp: int, sequence_name: str, save_root: str, img_metas: dict):
    """
    for EventStereoObjectDetectionNetwork.
    Args:
        pred:
            objdet: List[Tensor]. Each tensor is a (NumInstance, 18) shape.
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
        with open(os.path.join(path_det_results_folder, str(indexBatch * batch_size + indexInBatch).zfill(6) + ".txt"), detresults_openmode) as detresult_file:
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
    imgHeight, imgWidth = img_metas['h_cam'], img_metas['w_cam']
    facets_info_batch = []
    for indexInBatch, detection in enumerate(pred['objdet']):
        left_bboxes = detection[:, 0:4].cpu().numpy()
        tl_x = numpy.clip(left_bboxes[:, 0], 0, imgWidth)
        tl_y = numpy.clip(left_bboxes[:, 1], 0, imgHeight)
        br_x = numpy.clip(left_bboxes[:, 2], 0, imgWidth)
        br_y = numpy.clip(left_bboxes[:, 3], 0, imgHeight)
        left_bboxes = numpy.stack([tl_x, tl_y, br_x, br_y], axis=1)
        right_bboxes = detection[:, 4:8].cpu().numpy()
        tl_x_r = numpy.clip(right_bboxes[:, 0], 0, imgWidth)
        br_x_r = numpy.clip(right_bboxes[:, 2], 0, imgWidth)
        right_bboxes = numpy.stack([tl_x_r, br_x_r], axis=1)
        sbboxes = numpy.concatenate([left_bboxes, right_bboxes], axis=-1)
        classes = detection[:, -2].cpu().numpy().astype('int')
        confidences = detection[:, -1].cpu().numpy()
        if detection.shape[-1] > 11:
            keypts_left = detection[:, 8:12].cpu().numpy()
            keypts_right = detection[:, 12:16].cpu().numpy()
            visz_left, visz_right = DrawResultBboxesAndKeyptsOnStereoEventFrame(
                pred['concentrate']['left'].squeeze().cpu().numpy()[:img_metas['h_cam'], :img_metas['w_cam']],
                pred['concentrate']['right'].squeeze().cpu().numpy()[:img_metas['h_cam'], :img_metas["w_cam"]],
                sbboxes,
                classes,
                confidences,
                keypts_left,
                keypts_right)
        visz = numpy.concatenate([visz_left, visz_right], axis=-2)
        visz = cv2.cvtColor(visz, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(path_det_visz_folder, str(indexBatch * batch_size + indexInBatch).zfill(6) + ".png"), visz)
        left_concentrated = pred['concentrate']['left'].squeeze().cpu().numpy()[:img_metas['h_cam'], :img_metas['w_cam']]
        left_concentrated = left_concentrated - left_concentrated.min()
        left_concentrated = (left_concentrated * 255 / left_concentrated.max()).astype('uint8')
        right_concentrated = pred['concentrate']['right'].squeeze().cpu().numpy()[:img_metas['h_cam'], :img_metas['w_cam']]
        right_concentrated = right_concentrated - right_concentrated.min()
        right_concentrated = (right_concentrated * 255 / right_concentrated.max()).astype('uint8')
        cv2.imwrite(os.path.join(path_concentrate_left_folder, str(indexBatch * batch_size + indexInBatch).zfill(6) + ".png"), left_concentrated)
        cv2.imwrite(os.path.join(path_concentrate_right_folder, str(indexBatch * batch_size + indexInBatch).zfill(6) + ".png"), right_concentrated)

    return
