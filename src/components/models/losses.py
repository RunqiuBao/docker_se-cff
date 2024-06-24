import torch.nn as nn
import torch.nn.functional as F
import torch
from .warp import disp_warp
import torch.distributed as dist


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

            if self.logger is not None:
                leftimage_view = left_img[0].detach().squeeze().cpu()
                leftimage_view -= leftimage_view.min()
                leftimage_view /= leftimage_view.max()
                rightimage_view = right_img[0].detach().cpu()
                rightimage_view -= rightimage_view.min()
                rightimage_view /= rightimage_view.max()
                if self.disp_loss_cfg["use_warp_loss"]:
                    leftimage_warped_view = left_img_warped[0].squeeze().detach().cpu()
                    leftimage_warped_view -= leftimage_warped_view.min()
                    leftimage_warped_view /= leftimage_warped_view.max()
                    self.logger.add_image(
                        "left sharp warped",
                        leftimage_warped_view
                    )
                self.logger.add_image(
                    "left sharp image",
                    leftimage_view
                )
                self.logger.add_image(
                    "right sharp image",
                    rightimage_view
                )

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
