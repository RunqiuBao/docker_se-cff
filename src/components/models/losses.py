import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy
import torchvision
import torch.distributed as dist
from torch import Tensor

from .yolo_pose_utils import TaskAlignedAssigner, dist2bbox, make_anchors, bbox_iou, bbox2dist, xywh2xyxy, xyxy2xywh
from .warp import disp_warp


# OKS_SIGMA = (
#     numpy.array([0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72, 0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89])
#     / 10.0
# )
OKS_SIGMA = (
    numpy.array([0.26, 0.25])
    / 10.0
)


class DisparityLoss(nn.Module):

    def __init__(self, disp_loss_cfg: dict, is_distributed=False, logger=None):
        super(DisparityLoss, self).__init__()
        self._smoothL1 = nn.SmoothL1Loss(reduction="none")
        self._warpLoss = nn.SmoothL1Loss(reduction="none")
        self.is_distributed = is_distributed
        self.disp_loss_cfg = disp_loss_cfg
        self.logger = logger

    def forward(self, x):
        pyramid_weight = [1/3, 2/3, 1.0, 1.0, 1.0]

        pred_disparity_pyramid, gt_disparity, left_img, right_img = x
        l1_loss_final = 0.0
        if self.disp_loss_cfg["use_warp_loss"]:
            warp_loss_final = 0.0
        mask = gt_disparity > 0
        for idx in range(len(pred_disparity_pyramid)):
            pred_disp = pred_disparity_pyramid[idx]
            weight = pyramid_weight[idx]

            if pred_disp.size(-1) != gt_disparity.size(-1):
                pred_disp = pred_disp.unsqueeze(1)
                pred_disp = F.interpolate(
                    pred_disp,
                    size=(gt_disparity.size(-2), gt_disparity.size(-1)),
                    mode='bilinear',
                    align_corners=False
                ) * (gt_disparity.size(-1) / pred_disp.size(-1))
                pred_disp = pred_disp.squeeze(1)

            # L1 loss
            l1_loss = self._smoothL1(
                pred_disp[mask], gt_disparity[mask]
            )

            # stereo warping loss
            if self.disp_loss_cfg["use_warp_loss"]:
                left_img_warped, valid_mask = disp_warp(right_img, pred_disp.unsqueeze(1))
                valid_mask = valid_mask.to(torch.bool)
                warp_loss = self._warpLoss(left_img[valid_mask], left_img_warped[valid_mask])

            # if self.logger is not None:
            #     leftimage_view = left_img[0].detach().squeeze().cpu()
            #     leftimage_view -= leftimage_view.min()
            #     leftimage_view /= leftimage_view.max()
            #     rightimage_view = right_img[0].detach().cpu()
            #     rightimage_view -= rightimage_view.min()
            #     rightimage_view /= rightimage_view.max()
            #     if self.disp_loss_cfg["use_warp_loss"]:
            #         leftimage_warped_view = left_img_warped[0].squeeze().detach().cpu()
            #         leftimage_warped_view -= leftimage_warped_view.min()
            #         leftimage_warped_view /= leftimage_warped_view.max()
            #         self.logger.add_image(
            #             "left sharp warped",
            #             leftimage_warped_view
            #         )
            #     self.logger.add_image(
            #         "left sharp image",
            #         leftimage_view
            #     )
            #     self.logger.add_image(
            #         "right sharp image",
            #         rightimage_view
            #     )

            if self.is_distributed:
                loss_list = [l1_loss, warp_loss] if self.disp_loss_cfg["use_warp_loss"] else [l1_loss]
                for index_loss, one_loss in enumerate(loss_list):
                    world_size = torch.distributed.get_world_size()
                    tensor_list = [
                        torch.zeros([1], dtype=torch.int).cuda() for _ in range(world_size)
                    ]
                    cur_tensor = torch.tensor([one_loss.size(0)], dtype=torch.int).cuda()
                    dist.all_gather(tensor_list, cur_tensor)
                    total_point = torch.sum(torch.Tensor(tensor_list))
                    one_loss = one_loss.sum() / total_point * world_size  # Note: balance the valid points across processes.
                    if index_loss == 0:
                        l1_loss_final += weight * one_loss
                    else:
                        warp_loss_final += weight * one_loss
            else:
                l1_loss_final += weight * l1_loss.mean()
                if self.disp_loss_cfg["use_warp_loss"]:
                    warp_loss_final += weight * warp_loss.mean()

        losses_result = {
            "l1_loss": l1_loss_final * self.disp_loss_cfg['ls_loss_weight']
        }
        if self.disp_loss_cfg["use_warp_loss"]:
            losses_result["warp_loss"] = warp_loss_final * self.disp_loss_cfg['warp_loss_weight']
        return losses_result


class ObjdetLoss(nn.Module):

    def __init__(self, objdet_loss_cfg: dict, is_distributed=False):
        super(ObjdetLoss, self).__init__()
        pass

    def forward(self):
        pass


class DFLoss(nn.Module):
    """Criterion class for computing DFL losses during training."""

    def __init__(self, reg_max=16) -> None:
        """Initialize the DFL module."""
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist, target):
        """
        Return sum of left and right DFL losses.

        Distribution Focal Loss (DFL) proposed in Generalized Focal Loss
        https://ieeexplore.ieee.org/document/9792391
        """
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class BboxLoss(nn.Module):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max=16):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class v8DetectionLoss:
    """Criterion class for computing training losses."""

    def __init__(self, model, tal_topk=10):  # model must be de-paralleled
        """Initializes v8DetectionLoss with the model, defining model-related properties and BCE loss function."""
        device = next(model.parameters()).device  # get model device
        h = model.loss_cfg  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = torchvision.ops.box_convert(out[..., 1:5].mul_(scale_tensor), in_fmt="cxcywh", out_fmt="xyxy")
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        # dfl_conf = pred_distri.view(batch_size, -1, 4, self.reg_max).detach().softmax(-1)
        # dfl_conf = (dfl_conf.amax(-1).mean(-1) + dfl_conf.amax(-1).amin(-1)) / 2

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            # pred_scores.detach().sigmoid() * 0.8 + dfl_conf.unsqueeze(-1) * 0.2,
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )

        loss[0] *= self.hyp["bbox_loss_weight"]  # box gain
        loss[1] *= self.hyp["cls_loss_weight"]  # cls gain
        loss[2] *= self.hyp["dfl_loss_weight"]  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)


class KeypointLoss(nn.Module):
    """Criterion class for computing training losses."""

    def __init__(self, sigmas: Tensor = None, oks_sigmas: numpy.ndarray = None) -> None:
        """Initialize the KeypointLoss class."""
        super().__init__()
        if sigmas is None:
            assert oks_sigmas is not None
            self.oks_sigmas = oks_sigmas
            self.sigmas = None
        else:
            assert oks_sigmas is None
            self.sigmas = sigmas

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        """Calculates keypoint loss factor and Euclidean distance loss for predicted and actual keypoints."""
        if self.sigmas is None:
            self.sigmas = torch.from_numpy(self.oks_sigmas).to(pred_kpts.device)
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)  # from cocoeval
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()


def kpts_decode(anchor_points, pred_kpts):
    """Decodes predicted keypoints to image coordinates."""
    y = pred_kpts.clone()
    y[..., :2] *= 2.0
    y[..., 0] += anchor_points[..., [0]] - 0.5
    y[..., 1] += anchor_points[..., [1]] - 0.5
    return y

def calculate_keypoints_loss(
    masks, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts, calculator_bce_pose_loss, calculator_keypoint_loss
):
    """
    Calculate the keypoints loss for the model.

    This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
    based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
    a binary classification loss that classifies whether a keypoint is present or not.

    Args:
        masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
        target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
        keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
        batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
        stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1) or (BS, N_anchors, 1).
        target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
        pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

    Returns:
        kpts_loss (torch.Tensor): The keypoints loss.
        kpts_obj_loss (torch.Tensor): The keypoints object loss.
    """
    
    batch_idx = batch_idx.flatten()
    batch_size = len(masks)

    # Find the maximum number of keypoints in a single image
    max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()

    # Create a tensor to hold batched keypoints
    batched_keypoints = torch.zeros(
        (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]), device=keypoints.device
    )

    # TODO: any idea how to vectorize this?
    # Fill batched_keypoints with keypoints based on batch_idx
    for i in range(batch_size):
        keypoints_i = keypoints[batch_idx == i]
        batched_keypoints[i, : keypoints_i.shape[0]] = keypoints_i

    # Expand dimensions of target_gt_idx to match the shape of batched_keypoints
    target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)

    # Use target_gt_idx_expanded to select keypoints from batched_keypoints
    selected_keypoints = batched_keypoints.gather(
        1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2])
    )

    # Divide coordinates by stride
    if len(stride_tensor.shape) == 2:
        selected_keypoints /= stride_tensor.view(1, -1, 1, 1)
    else:
        selected_keypoints /= stride_tensor.view(batch_size, -1, 1, 1)

    kpts_loss = 0
    kpts_obj_loss = 0

    if masks.any():
        gt_kpt = selected_keypoints[masks]
        area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
        pred_kpt = pred_kpts[masks]
        kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
        kpts_loss = calculator_keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

        if pred_kpt.shape[-1] == 3:
            kpts_obj_loss = calculator_bce_pose_loss(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

    return kpts_loss, kpts_obj_loss


class v8PoseLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes v8PoseLoss with model, sets keypoint variables and declares a keypoint loss instance."""
        super().__init__(model)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [2, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def __call__(self, preds, batch):
        """Calculate the total loss and detach it."""
        loss = torch.zeros(5, device=self.device)
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        batch_size = pred_scores.shape[0]
        batch_idx = batch["batch_idx"].view(-1, 1)
        targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        pred_kpts = kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, numKeypts, 3)

        target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        batch_target_labels, batch_target_bboxes, batch_target_scores, batch_target_gt_idx = [], [], [], []
        for indexInBatch in range(batch_size):
            batch_target_bboxes.append(target_bboxes[indexInBatch][fg_mask[indexInBatch]])
            batch_target_scores.append(target_scores[indexInBatch][fg_mask[indexInBatch]])
            batch_target_gt_idx.append(target_gt_idx[indexInBatch][fg_mask[indexInBatch]])
            batch_target_labels.append(target_labels[indexInBatch][fg_mask[indexInBatch]])

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[3] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[4] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            loss[1], loss[2] = calculate_keypoints_loss(
                fg_mask,
                target_gt_idx,
                keypoints,
                batch_idx,
                stride_tensor,
                target_bboxes,
                pred_kpts,
                self.bce_pose,
                self.keypoint_loss
            )
        loss[0] *= self.hyp["bbox_loss_weight"]  # box gain
        loss[1] *= self.hyp["pose_loss_weight"]  # pose gain
        loss[2] *= self.hyp["kobj_loss_weight"]  # kobj gain
        loss[3] *= self.hyp["cls_loss_weight"]  # cls gain
        loss[4] *= self.hyp["dfl_loss_weight"]  # dfl gain
        
        selected_bboxes = (pred_bboxes.detach().clone() * stride_tensor)[fg_mask]
        pred_kpts_clone = pred_kpts.detach().clone()[..., :2]
        stride_tensor_expand = (stride_tensor.unsqueeze(0).unsqueeze(-1)).expand(batch_size, -1, 2, 2)
        selected_keypts = (pred_kpts_clone * stride_tensor_expand)[fg_mask]
        selected_confidences, selected_cls= torch.max(pred_scores[fg_mask].detach().clone(), dim=1)
        device = pred_bboxes.device
        num_instances = pred_bboxes.shape[1]
        selected_batchidx = torch.arange(batch_size, dtype=torch.int, device=device).unsqueeze(1).expand(batch_size, num_instances)[fg_mask]
        return {"loss_leftdetection": loss.sum() * batch_size}, selected_bboxes, selected_cls, selected_confidences, selected_keypts, selected_batchidx, fg_mask, target_gt_idx
