import torch.nn as nn
import torch
from torch import Tensor
import numpy
from typing import List, Dict, Tuple
from thop import profile
import cv2
import time

from .concentration import ConcentrationNet
from .stereo_matching import StereoMatchingNetwork
from .objectdetection import Cylinder5DDetectionHead
from .feature_extractor import FeatureExtractor2
from .utils.misc import multi_apply, timeit

from . import losses
from .utils.misc import freeze_module_grads


class EventStereoObjectDetectionNetwork(nn.Module):

    def __init__(
        self,
        concentration_net_cfg: dict = None,
        feature_extraction_net_cfg: dict = None,
        disp_head_cfg: dict = None,
        object_detection_head_cfg: dict = None,
        losses_cfg: dict = None,  # Cylinder5DDetectionLoss, disparityLoss
        is_distributed: bool = False,
        logger=None,
    ):
        super(EventStereoObjectDetectionNetwork, self).__init__()
        self.logger = logger
        self.is_freeze_disp = disp_head_cfg['is_freeze']  # Note: when training disparity, skip object detection.
        # ==========  concentration net ===========
        self._concentration_net = ConcentrationNet(**concentration_net_cfg.PARAMS)
        if self.is_freeze_disp:
            freeze_module_grads(self._concentration_net)
        # =========== feature extractor ===========
        self._feature_extraction_net = FeatureExtractor2(
            net_cfg=feature_extraction_net_cfg.PARAMS
        )
        if not self.is_freeze_disp:
            freeze_module_grads(self._feature_extraction_net)
        # ============ stereo matching net ============
        self._disp_head = StereoMatchingNetwork(
            **disp_head_cfg.PARAMS, isInputFeature=False  # Note: an efficient feature extractor for object detection might not be good for stereo matching?
        )
        if self.is_freeze_disp:
            freeze_module_grads(self._disp_head)
        # ============= object detection net =============
        self._object_detection_head = Cylinder5DDetectionHead(
            net_cfg=object_detection_head_cfg.PARAMS,
            loss_cfg=losses_cfg['objdet_loss_cfg'],
            is_distributed=is_distributed,
            logger=logger
        )
        if not self.is_freeze_disp:
            freeze_module_grads(self._object_detection_head)
        # ============= losses ============
        if not self.is_freeze_disp:
            self._disp_loss = getattr(losses, losses_cfg['disp_loss_cfg']['NAME'])(
                losses_cfg['disp_loss_cfg'],
                is_distributed=is_distributed,
                logger=logger
            )

    def forward(self, left_event: Tensor, right_event: Tensor, gt_labels: list[dict] = None, batch_img_metas: dict = None):
        """
        Args:
            left/right_event: (b c s t h w) tensor
            gt_labels: a list of dict, each dict contains 'disparity' and 'objdet'.
                       'disparity' contains Tensor; 'objdet' is another dict contains 'bboxes' and 'labels'.
            batch_img_metas: dict. It contains info about image height and width.
        """
        if self.logger is not None:
            event_view = left_event[0, 0, :, :, 0, 0].detach().cpu()
            event_view -= event_view.min()
            event_view /= event_view.max()
            self.logger.add_image(
                "left event input",
                event_view
            )
        left_event = left_event.squeeze(1).squeeze(3).permute(0, 3, 1, 2)
        right_event = right_event.squeeze(1).squeeze(3).permute(0, 3, 1, 2)

        left_event_sharp = self._concentration_net(left_event)
        right_event_sharp = self._concentration_net(right_event)

        # FIXME: test using same feature extractor for both head
        pred_disparity_pyramid = self._disp_head(left_event_sharp, right_event_sharp)

        loss_final = None
        preds_final = {}
        if not self.is_freeze_disp and gt_labels is not None:
            loss_final = self._disp_loss((pred_disparity_pyramid, gt_labels['disparity'], left_event_sharp, right_event_sharp))
        if self.is_freeze_disp:
            left_feature = self._feature_extraction_net(left_event_sharp.repeat(1, 3, 1, 1))
            right_feature = self._feature_extraction_net(right_event_sharp.repeat(1, 3, 1, 1))
            object_preds, loss_final = self._object_detection_head(
                left_feature,
                right_feature,
                pred_disparity_pyramid[-1],  # use full size disparity prediction as prior to help stereo detection
                batch_img_metas,
                gt_labels["objdet"]
            )

            preds_final['objdet'] = object_preds
            if self.logger is not None:
                if 'sbboxes' in object_preds[0]:
                    leftimage_views, rightimage_views = multi_apply(
                        self.RenderImageWithBboxesAndKeypts,
                        left_event_sharp.detach().squeeze(1).cpu().numpy(),
                        right_event_sharp.detach().squeeze(1).cpu().numpy(),
                        object_preds,
                    )
                    self.logger.add_image(
                        "left sharp with bboxes, keypoints",
                        leftimage_views[0]
                    )
                    self.logger.add_image(
                        "right sharp with bboxes",
                        rightimage_views[0]
                    )
                else:
                    leftimage_views = []
                    num_imgs = left_event.shape[0]
                    for i in range(num_imgs):
                        leftimage_views.append(
                            self.RenderImageWithBboxes(left_event_sharp.detach().squeeze(1).cpu().numpy()[i], object_preds[i])
                        )
                    self.logger.add_image(
                        "left sharp with bboxes, keypoints",
                        leftimage_views[0]
                    )

        preds_final['disparity'] = pred_disparity_pyramid[-1]
        torch.cuda.synchronize()
        return preds_final, loss_final

    def get_params_group(self, learning_rate):
        def filter_specific_params(kv):
            specific_layer_name = ["offset_conv.weight", "offset_conv.bias"]  # Note: exist in deform conv.
            for name in specific_layer_name:
                if name in kv[0]:
                    return True
            return False

        def filter_base_params(kv):
            specific_layer_name = ["offset_conv.weight", "offset_conv.bias"]
            for name in specific_layer_name:
                if name in kv[0]:
                    return False
            return True

        specific_params = list(filter(filter_specific_params, self.named_parameters()))
        base_params = list(filter(filter_base_params, self.named_parameters()))

        specific_params = [
            kv[1] for kv in specific_params
        ]  # kv is a tuple (key, value)
        base_params = [kv[1] for kv in base_params]

        specific_lr = learning_rate * 0.1
        params_group = [
            {"params": base_params, "lr": learning_rate},
            {"params": specific_params, "lr": specific_lr},
        ]

        return params_group

    @staticmethod
    def ComputeCostProfile(model, inputShape):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        left_event = torch.randn(*inputShape).to(device)
        right_event = torch.randn(*inputShape).to(device)
        model = model.to(device)
        flops, numParams = profile(model, inputs=(left_event, right_event), verbose=False)
        return flops, numParams

    def RenderImageWithBboxesAndKeypts(
        self,
        left_event_sharp: numpy.ndarray,
        right_event_sharp: numpy.ndarray,
        obj_preds: Dict
    ) -> Tuple[Tensor]:
        """
        Args:
            left_event_sharp: ...
        """
        sbboxes, classes, confidences, keypts1, keypts2 = obj_preds['sbboxes'], obj_preds['classes'], obj_preds['confidences'], obj_preds['keypt1s'], obj_preds['keypt2s']
        if keypts1.ndim == 1:
            keypts1 = keypts1.unsqueeze(0)
            keypts2 = keypts2.unsqueeze(0)
        left_event_sharp = left_event_sharp - left_event_sharp.min()
        left_event_sharp = (left_event_sharp * 255 / left_event_sharp.max()).astype('uint8')
        left_event_sharp = cv2.cvtColor(left_event_sharp, cv2.COLOR_GRAY2RGB)
        right_event_sharp = right_event_sharp - right_event_sharp.min()
        right_event_sharp = (right_event_sharp * 255 / right_event_sharp.max()).astype('uint8')
        right_event_sharp = cv2.cvtColor(right_event_sharp, cv2.COLOR_GRAY2RGB)
        for bbox, classindex, confidence, keypt1, keypt2 in zip(sbboxes, classes, confidences, keypts1, keypts2):
            top_left = (int(bbox[0]), int(bbox[1]))
            top_right = (int(bbox[2]), int(bbox[1]))
            bottom_right = (int(bbox[2]), int(bbox[3]))
            cv2.rectangle(left_event_sharp, top_left, bottom_right, (255, 0, 0), thickness=3)
            text = 'cls: {}\nconfi: {}'.format(format(classindex.item(), '.2f'), format(confidence.item(), '.2f'))
            textposition = (int(top_right[0] + bottom_right[1]) // 2, int(top_right[1] + bottom_right[1]) // 2)
            cv2.putText(left_event_sharp, text, textposition, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 255, 0))
            w, h = bottom_right[0] - top_left[0], bottom_right[1] - top_right[1]
            keypt1_int = (int(keypt1[0] * w + top_left[0]), int(keypt1[1] * h + top_left[1]))
            keypt2_int = (int(keypt2[0] * w + top_left[0]), int(keypt2[1] * h + top_left[1]))
            cv2.circle(left_event_sharp, keypt1_int, radius=3, color=(0, 255, 0), thickness=-1)
            cv2.circle(left_event_sharp, keypt2_int, radius=3, color=(0, 0, 255), thickness=-1)

            top_left = (int(bbox[4]), int(bbox[1]))
            bottom_right = (int(bbox[5]), int(bbox[3]))
            cv2.rectangle(right_event_sharp, top_left, bottom_right, (255, 0, 0), thickness=3)
        return torch.from_numpy(left_event_sharp), torch.from_numpy(right_event_sharp)

    def RenderImageWithBboxes(
        self,
        left_event_sharp: numpy.ndarray,
        obj_preds: Dict
    ) -> Tensor:
        """
        Args:
            left_event_sharp: ...
        """
        bboxes, classes, confidences = obj_preds['bboxes'], obj_preds['classes'], obj_preds['confidences']
        left_event_sharp = left_event_sharp - left_event_sharp.min()
        left_event_sharp = (left_event_sharp * 255 / left_event_sharp.max()).astype('uint8')
        left_event_sharp = cv2.cvtColor(left_event_sharp, cv2.COLOR_GRAY2RGB)
        for bbox, classindex, confidence in zip(bboxes, classes, confidences):
            top_left = (int(bbox[0]), int(bbox[1]))
            top_right = (int(bbox[2]), int(bbox[1]))
            bottom_right = (int(bbox[2]), int(bbox[3]))
            cv2.rectangle(left_event_sharp, top_left, bottom_right, (255, 0, 0), thickness=3)
            text = 'cls: {}\nconfi: {}'.format(format(classindex.item(), '.2f'), format(confidence.item(), '.2f'))
            textposition = (int(top_right[0] + bottom_right[1]) // 2, int(top_right[1] + bottom_right[1]) // 2)
            cv2.putText(left_event_sharp, text, textposition, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 255, 0))
            w, h = bottom_right[0] - top_left[0], bottom_right[1] - top_right[1]
        return [torch.from_numpy(left_event_sharp),]
