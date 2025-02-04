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


from . import losses
from .utils.misc import freeze_module_grads, unfreeze_module_grads, multi_apply, convert_tensor_to_numpy
from ..methods.visz_utils import RenderImageWithBboxes, RenderImageWithBboxesAndKeypts


class OnnxStyleNetwork(nn.Module):
    def __init__(self, model):
        super(OnnxStyleNetwork, self).__init__()
        self.model = model

    def forward(self, left_event: Tensor, right_event: Tensor, h_cam: Tensor, w_cam: Tensor):
        h_tensor, w_tensor = left_event.shape[-2:]
        batch_img_metas = {"h_cam": h_cam.item(), "w_cam": w_cam.item(), "h": h_tensor, "w": w_tensor}
        preds, losses = self.model(left_event, right_event, None, batch_img_metas)
        output_concentrate_left = preds["concentrate"]["left"]
        output_concentrate_right = preds["concentrate"]["right"]
        output_disparity = preds["disparity"][0]
        output_objdet = preds["objdet"][0]  # Note: should only use batch_size = 1 when doing inference.
        output_facets = preds["objdet_facets"][0]
        output_facets_right = preds["objdet_facets_right"][0]
        return output_objdet, output_facets, output_facets_right, output_concentrate_left, output_concentrate_right, output_disparity


class EventStereoObjectDetectionNetwork(nn.Module):

    def __init__(
        self,
        concentration_net_cfg: dict = None,
        feature_extraction_net_cfg: dict = None,
        disp_head_cfg: dict = None,
        object_detection_head_cfg: dict = None,
        losses_cfg: dict = None,  # Cylinder5DDetectionLoss, disparityLoss
        is_distributed: bool = False,
        is_test = False,
        logger=None,
    ):
        super(EventStereoObjectDetectionNetwork, self).__init__()
        self.is_test = False  # Note: whether is test only
        self.logger = logger
        self.is_freeze_disp = disp_head_cfg['is_freeze']  # Note: when training disparity, skip object detection.
        is_train_featmaponly = (
                object_detection_head_cfg['PARAMS']['keypt_pred_cfg']['is_enable'] and object_detection_head_cfg['PARAMS']['keypt_pred_cfg']['PARAMS']['is_train_keypt']
            ) or (
                object_detection_head_cfg['PARAMS']['facet_pred_cfg']['is_enable'] and object_detection_head_cfg['PARAMS']['facet_pred_cfg']['PARAMS']['is_train_facet']
            )
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
        if is_train_featmaponly:
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

    def SetTest(self):
        self.is_test = True
    
    def SetNotTest(self):
        self.is_test = False

    def forward(self, left_event: Tensor, right_event: Tensor, gt_labels: dict = None, batch_img_metas: dict = None, **kwargs):
        """
        Args:
            left/right_event: (b c s t h w) tensor
            gt_labels: a dict, each dict contains 'disparity' and 'objdet'.
                       'disparity' is a Tensor; 'objdet' is list of dict, each dict contains 'bboxes' and 'labels' and 'keypt_masks'.
            batch_img_metas: dict. It contains info about image height and width.
        """
        if self.logger is not None:
            event_view = left_event[0, -3, :, :].detach().cpu()
            event_view -= event_view.min()
            event_view /= event_view.max()
            self.logger.add_image(
                "left event input",
                event_view
            )

        starttime = time.time()        
        left_event_sharp = self._concentration_net(left_event)
        right_event_sharp = self._concentration_net(right_event)
        # print("time1: {}".format(time.time() - starttime))

        # FIXME: test using same feature extractor for both head
        starttime = time.time()
        pred_disparity_pyramid = self._disp_head(left_event_sharp, right_event_sharp)
        # print("time2: {}".format(time.time() - starttime))

        loss_final = None
        if not self.is_freeze_disp and len(gt_labels) > 0:
            try:
                loss_final = self._disp_loss((
                    pred_disparity_pyramid,
                    gt_labels['disparity'],
                    left_event_sharp,
                    right_event_sharp
                ))
            except:
                from IPython import embed; embed()

        if self.is_freeze_disp or len(gt_labels) == 0:
            starttime = time.time()
            left_feature = self._feature_extraction_net(left_event_sharp.repeat(1, 3, 1, 1))
            right_feature = self._feature_extraction_net(right_event_sharp.repeat(1, 3, 1, 1))
            # print("time3: {}".format(time.time() - starttime))

            starttime = time.time()
            object_preds, loss_final = self._object_detection_head(
                left_feature,
                right_feature,
                [left_event],
                [right_event],
                pred_disparity_pyramid[-1],  # use full size disparity prediction as prior to help stereo detection
                batch_img_metas,
                gt_labels["objdet"] if gt_labels is not None and len(gt_labels) != 0 else None
            )
            # print("time4: {}".format(time.time() - starttime))

            if self.logger is not None and len(object_preds) > 0 and loss_final is not None and (not self.is_test):
                if 'sbboxes' in object_preds[0]:                             
                    leftimage_views, rightimage_views = multi_apply(
                        RenderImageWithBboxesAndKeypts,
                        left_event_sharp.detach().squeeze(1).cpu().numpy(),
                        right_event_sharp.detach().squeeze(1).cpu().numpy(),
                        convert_tensor_to_numpy(object_preds)
                    )
                    self.logger.add_image(
                        "left sharp with bboxes",
                        leftimage_views[0]
                    )
                    self.logger.add_image(
                        "right sharp with bboxes",
                        rightimage_views[0]
                    )

                    # gt_objects_preds = []
                    # for gt_objects in gt_labels["objdet"]:
                    #     gt_objects_preds.append({
                    #         "sbboxes": gt_objects["bboxes"][:, :6].cpu(),
                    #         "classes": gt_objects["labels"].cpu(),
                    #         "confidences": torch.ones((gt_objects["labels"].shape[0],), dtype=gt_objects["labels"].dtype, device='cpu')
                    #     })
                    # leftimage_withGT_views, rightimage_withGT_views = multi_apply(
                    #     RenderImageWithBboxesAndKeypts,
                    #     left_event_sharp.detach().squeeze(1).cpu().numpy(),
                    #     right_event_sharp.detach().squeeze(1).cpu().numpy(),
                    #     gt_objects_preds
                    # )
                    # self.logger.add_image(
                    #     "left sharp with GT bboxes",
                    #     leftimage_withGT_views[0]
                    # )
                    # self.logger.add_image(
                    #     "right sharp with GT bboxes",
                    #     rightimage_withGT_views[0]
                    # )
                else:
                    leftimage_views = []
                    num_imgs = left_event.shape[0]
                    for i in range(num_imgs):
                        leftimage_views.append(
                            RenderImageWithBboxes(left_event_sharp.detach().squeeze(1).cpu().numpy()[i], object_preds[i])
                        )
                    self.logger.add_image(
                        "left sharp with bboxes, keypoints",
                        leftimage_views[0]
                    )
            if len(object_preds) == 0:
                print("Zero detection occured.")

        # Prepare inference output.
        preds_final = {}
        # split objdet and objdet_facets if objdet_facets exist.
        if self.is_test:
            if len(object_preds) == 0:
                preds_final['objdet'] = object_preds
            # elif isinstance(object_preds[0], torch.Tensor):
            #     preds_final['objdet'] = [pred.detach().cpu() for pred in object_preds]
            elif isinstance(object_preds[0], dict):
                preds_final['objdet_facets'] = [detection.pop('facets').detach().cpu() for detection in object_preds]
                preds_final['objdet_facets_right'] = [detection.pop('facets_right').detach().cpu() for detection in object_preds]
                preds_final['enlarge_facet_factor'] = [detection.pop('enlarge_facet_factor') for detection in object_preds]
                preds_final['objdet'] = [detection.pop('detection').detach().cpu() for detection in object_preds]
            preds_final['concentrate'] = {
                'left': left_event_sharp.detach().cpu(),
                'right': right_event_sharp.detach().cpu()
            }
        preds_final['disparity'] = pred_disparity_pyramid[-1].detach().cpu()

        torch.cuda.synchronize()

        if self.logger is not None:
            lsharp_view = left_event_sharp[0, 0, :, :,].detach().cpu()
            lsharp_view -= lsharp_view.min()
            lsharp_view /= lsharp_view.max()
            self.logger.add_image(
                "left sharp",
                lsharp_view
            )
            disparity_view = pred_disparity_pyramid[-1].detach().cpu()
            disparity_view -= disparity_view.min()
            disparity_view /= disparity_view.max()
            self.logger.add_image(
                "disparity view",
                disparity_view
            )
            if gt_labels is not None and 'disparity' in gt_labels:
                disparity_gt = gt_labels['disparity'].detach().squeeze().cpu()
                disparity_gt -= disparity_gt.min()
                disparity_gt /= disparity_gt.max()
                self.logger.add_image(
                    "disparity gt",
                    disparity_gt
                )
        return preds_final, loss_final

    def get_params_group(self, learning_rate, keypt_lr=None):
        if keypt_lr is not None:
            specific_layer_name = ['_keypt_feature_extraction_net', 'keypt2_predictor', 'keypt1_predictor', "offset_conv.weight", "offset_conv.bias"]
        else:
            specific_layer_name = ["offset_conv.weight", "offset_conv.bias"]# Note: exist in deform conv.
        def filter_specific_params(kv):
            specific_layer_name = ["offset_conv.weight", "offset_conv.bias"]  
            for name in specific_layer_name:
                if name in kv[0]:
                    return True
            return False

        keypt_layer_name = ["keypt1_predictor", "keypt2_predictor"]
        def filter_keypt_params(kv):
            for name in keypt_layer_name:
                if name in kv[0]:
                    return True
            return False

        all_specific_layer_name = ["offset_conv.weight", "offset_conv.bias"]
        if keypt_lr is not None:
            all_specific_layer_name.extend(keypt_layer_name)
        def filter_base_params(kv):
            for name in all_specific_layer_name:
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
        if keypt_lr is None:
            params_group = [
                {"params": base_params, "lr": learning_rate},
                {"params": specific_params, "lr": specific_lr},
            ]
        else:
            keypt_params = list(filter(filter_keypt_params, self.named_parameters()))
            keypt_params = [kv[1] for kv in keypt_params]
            params_group = [
                {"params": base_params, "lr": learning_rate},
                {"params": specific_params, "lr": specific_lr},
                {"params": keypt_params, "lr": keypt_lr}
            ]   
        return params_group

    @staticmethod
    def ComputeCostProfile(model, inputShape):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        left_event = torch.randn(*inputShape).to(device)
        right_event = torch.randn(*inputShape).to(device)
        model = model.to(device)
        
        flops, numParams = profile(model, inputs=(left_event, right_event, {}, {'h': inputShape[-2], 'w': inputShape[-1]}), verbose=False)
        return flops, numParams

