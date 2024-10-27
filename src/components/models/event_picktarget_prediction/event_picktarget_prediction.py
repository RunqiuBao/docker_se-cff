import torch.nn as nn
import torch
from torch import Tensor
import torchvision
from typing import List, Dict, Tuple
from thop import profile
import copy
import time

from .rtdetr import RTDETR
from ..concentration import ConcentrationNet
from .rtdetr_criterion import RTDETRCriterion
from ..utils.misc import freeze_module_grads
from .rtdetr_criterion import RTDETRCriterion
from .matcher import HungarianMatcher

from ...methods.visz_utils import RenderImageWithBboxes


class EventPickTargetPredictionNetwork(nn.Module):
    def __init__(
        self,
        concentration_net_cfg: dict,
        rtdetr_cfg: dict,
        losses_cfg: dict,
        is_distributed: bool = False,
        logger = None
    ):
        super(EventPickTargetPredictionNetwork, self).__init__()
        self._concentration_net = ConcentrationNet(**concentration_net_cfg.PARAMS)
        freeze_module_grads(self._concentration_net)
        self._rtdetr = RTDETR(rtdetr_cfg=rtdetr_cfg)
        # prepare the loss function
        matcher = HungarianMatcher(**losses_cfg.matcher.params)
        self.loss_functor = RTDETRCriterion(matcher, **losses_cfg.params)
        self.logger = logger

    def forward(self, left_event: Tensor, right_event: Tensor, gt_labels: Dict=None, batch_img_metas: Dict=None, global_step_info: Dict=None, is_train: bool=True):
        results = {}
        list_event_sharp = []
        for x, side in zip([left_event, right_event], ["left", "right"]):
            x = x.squeeze(1).squeeze(3).permute(0, 3, 1, 2)
            sharp_repr = self._concentration_net(x)
            list_event_sharp.append(sharp_repr.detach().squeeze(1).cpu().numpy())
            results[side] = self._rtdetr(sharp_repr.repeat(1, 3, 1, 1), targets=gt_labels['objdet'][side] if gt_labels is not None else None)
            if gt_labels is None:
                results[side + "_sharprepr"] = sharp_repr.detach().cpu().numpy()
        loss_final = None
        if is_train:
            self.loss_functor.train()
        else:
            self.loss_functor.eval()

        if gt_labels is not None:
            assert (global_step_info is not None)
            epoch, lengthDataLoader, indexBatch = global_step_info["epoch"], global_step_info["lengthDataLoader"], global_step_info["indexBatch"]
            global_step = epoch * lengthDataLoader + indexBatch
            epoch_info = dict(epoch=epoch, step=indexBatch, global_step=global_step)
            losses = []
            predictions_leftright = {}
            for indexResult, result in enumerate(results.values()):
                loss, predictions = self.loss_functor(result, gt_labels['objdet']["left" if indexResult == 0 else "right"], **epoch_info)
                losses.extend(loss.values())                
                predictions_leftright["left" if indexResult == 0 else "right"] = predictions  # NOte: list, for batch inputs

            loss_final = {
                "Loss": sum(losses)
            }

            if self.logger is not None and len(predictions_leftright) > 0:                
                predictions_left = copy.deepcopy(predictions_leftright["left"])[0]
                predictions_left["bboxes"] = torchvision.ops.box_convert(predictions_left["bboxes"], in_fmt="cxcywh", out_fmt="xyxy".lower())
                predictions_left["bboxes"][:, [0, 2]] *= batch_img_metas['w']
                predictions_left["bboxes"][:, [1, 3]] *= batch_img_metas['h']                                
                view_img = RenderImageWithBboxes(list_event_sharp[0][0], predictions_left)
                self.logger.add_image(
                    "left sharp with bboxes",
                    view_img[0]
                )
        else:
            for side in ["left", "right"]:
                mask_pred_logits_objects = torch.max(results[side]["pred_logits"], dim=2)[0] > 0
                results[side + "_detections"] = results[side]["pred_boxes"][mask_pred_logits_objects]
                results[side + "_classes"] = torch.max(results[side]["pred_logits"], dim=2)[0][mask_pred_logits_objects]

        return results, loss_final

    @staticmethod
    def ComputeCostProfile(model: torch.nn.Module, inputShape: list):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        left_event = torch.randn(*inputShape).to(device)
        right_event = torch.randn(*inputShape).to(device)
        model = model.to(device)        
        flops, numParams = profile(model, inputs=(left_event, right_event, None, {'h': inputShape[0], 'w': inputShape[1]}), verbose=False)        
        return flops, numParams
