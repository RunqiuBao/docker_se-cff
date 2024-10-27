import torch.nn as nn
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from torch import Tensor
from thop import profile
import time


from .concentration import ConcentrationNet
from .utils.misc import freeze_module_grads, unfreeze_module_grads


class EventStereoSegmentationNetwork(nn.Module):
    _logger = None

    def __init__(
        self,
        concentration_net_cfg: dict,
        seg_net_cfg: dict,
        losses_cfg: dict = None,
        is_distributed: bool = False,
        logger = None
    ):
        super(EventStereoSegmentationNetwork, self).__init__()
        self._concentration_net = ConcentrationNet(**concentration_net_cfg.PARAMS)
        freeze_module_grads(self._concentration_net)  # Note: preload concentration_net weights from dsec.
        self._mobile_unet = smp.Unet('timm-mobilenetv3_large_100', in_channels=seg_net_cfg.PARAMS.in_channels)
        self.loss_l1 = torch.nn.SmoothL1Loss(reduction='mean')
        if logger is not None:
            self._logger = logger

    def forward(self, left_event: Tensor, right_event: Tensor, gt_labels: dict = None, batch_img_metas: dict = None):
        """
        Args:
            left_event, right_event: accumulated event stacks.
            gt_labels: 'objdet' contains 'left' and 'right' keys, each contains a GT target segmentation map.
            batch_img_metas: include 'h', 'w', 'h_cam', 'w_cam' keys.
        """
        left_event = left_event.squeeze(1).squeeze(3).permute(0, 3, 1, 2)
        right_event = right_event.squeeze(1).squeeze(3).permute(0, 3, 1, 2)

        starttime = time.time()
        left_event_sharp = self._concentration_net(left_event)
        right_event_sharp = self._concentration_net(right_event)
        # print("time1: {}".format(time.time() - starttime))

        starttime = time.time()
        probability_map_left = self._mobile_unet(left_event_sharp)
        probability_map_right = self._mobile_unet(right_event_sharp)
        # print("time2: {}".format(time.time() - starttime))
        if self._logger is not None:
            probability_map_left_view = probability_map_left.detach().cpu()
            probability_map_left_view -= probability_map_left_view.min()
            probability_map_left_view /= probability_map_left_view.max()            
            self._logger.add_image("probability_map_left", probability_map_left_view.squeeze())
            left_event_sharp_view = left_event_sharp.detach().cpu()
            left_event_sharp_view -= left_event_sharp_view.min()
            left_event_sharp_view /= left_event_sharp_view.max()
            self._logger.add_image("left_event_sharp", left_event_sharp_view.squeeze())

        loss_pmap = None        
        if gt_labels is not None and len(gt_labels) > 0:                        
            loss_pmap = self.loss_l1(probability_map_left, gt_labels["objdet"][0]["pmap"]["left"]) + self.loss_l1(probability_map_right, gt_labels["objdet"][0]["pmap"]["right"])
            if self._logger is not None:
                self._logger.add_image("gt_labels", gt_labels["objdet"][0]["pmap"]["left"].detach().cpu())

        return {"left_pmap": probability_map_left, "right_pmap": probability_map_right, "left_concentrate": left_event_sharp, "right_concentrate": right_event_sharp}, {"loss_pmap": loss_pmap}

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
        flops, numParams = profile(model, inputs=(left_event, right_event, None, {'h': inputShape[0], 'w': inputShape[1]}), verbose=False)
        return flops, numParams
