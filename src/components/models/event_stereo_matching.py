import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy
import cv2
import onnxruntime

from einops import rearrange

from .concentration import ConcentrationNet
from .stereo_matching import StereoMatchingNetwork


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


class EventStereoMatchingNetwork(nn.Module):
    count = 0
    onnx_program = False

    def __init__(self, 
                 concentration_net=None,
                 disparity_estimator=None):
        super(EventStereoMatchingNetwork, self).__init__()
        self.concentration_net = ConcentrationNet(**concentration_net.PARAMS)
        self.stereo_matching_net = StereoMatchingNetwork(**disparity_estimator.PARAMS)

        self.criterion = nn.SmoothL1Loss(reduction='none')

    def forward(self, left_event, right_event, gt_disparity=None):
        event_stack = {
            'l': left_event.clone(),
            'r': right_event.clone(),
        }
        
        concentrated_event_stack = {}
        for loc in ['l', 'r']:            
            event_stack[loc] = rearrange(event_stack[loc], 'b c h w t s -> b (c s t) h w')
            concentrated_event_stack[loc] = self.concentration_net(event_stack[loc])
        
        # if not self.onnx_program:
        #     torch.onnx.export(
        #         self.concentration_net,
        #         event_stack['l'][0].unsqueeze(0),
        #         "/home/runqiu/code/se-cff/concentrate_events.onnx",
        #         export_params=True,
        #         opset_version=16,
        #         do_constant_folding=True,
        #         input_names=["input"],
        #         output_names=["output"],
        #         dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
        #     )
        #     self.onnx_program = True
        # else:
        #     # providers=["CPUExecutionProvider"]
        #     providers = [("CUDAExecutionProvider", {"device_id": torch.cuda.current_device()})]
        #     ort_session = onnxruntime.InferenceSession("/home/runqiu/code/se-cff/concentrate_events.onnx", providers=providers)
        #     aa = concentrated_event_stack['l']
        #     for indexSlice in range(aa.shape[0]):
        #         ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(event_stack['l'][indexSlice].unsqueeze(0))}
        #         ort_outs = ort_session.run(None, ort_inputs) 
        #         oneImg = numpy.squeeze(ort_outs)
        #         oneImg -= oneImg.min()
        #         oneImg *= 255 / oneImg.max()
        #         cv2.imwrite("/media/runqiu/HDD1/opensource-dataset/dsec/concentrated/" + str(self.count).zfill(6) + ".png", oneImg.astype(numpy.uint8))
        #         self.count += 1

        pred_disparity_pyramid = self.stereo_matching_net(
            concentrated_event_stack['l'],
            concentrated_event_stack['r']
        )

        loss_disp = None
        if gt_disparity is not None:
            loss_disp = self._cal_loss(pred_disparity_pyramid, gt_disparity)

        return pred_disparity_pyramid[-1], loss_disp

    def get_params_group(self, learning_rate):
        def filter_specific_params(kv):
            specific_layer_name = ['offset_conv.weight', 'offset_conv.bias']
            for name in specific_layer_name:
                if name in kv[0]:
                    return True
            return False

        def filter_base_params(kv):
            specific_layer_name = ['offset_conv.weight', 'offset_conv.bias']
            for name in specific_layer_name:
                if name in kv[0]:
                    return False
            return True

        specific_params = list(filter(filter_specific_params,
                                      self.named_parameters()))
        base_params = list(filter(filter_base_params,
                                  self.named_parameters()))

        specific_params = [kv[1] for kv in specific_params]  # kv is a tuple (key, value)
        base_params = [kv[1] for kv in base_params]

        specific_lr = learning_rate * 0.1
        params_group = [
            {'params': base_params, 'lr': learning_rate},
            {'params': specific_params, 'lr': specific_lr},
        ]

        return params_group

    def _cal_loss(self, pred_disparity_pyramid, gt_disparity):
        pyramid_weight = [1 / 3, 2 / 3, 1.0, 1.0, 1.0]

        loss = 0.0
        mask = gt_disparity > 0
        for idx in range(len(pred_disparity_pyramid)):
            pred_disp = pred_disparity_pyramid[idx]
            weight = pyramid_weight[idx]

            if pred_disp.size(-1) != gt_disparity.size(-1):
                pred_disp = pred_disp.unsqueeze(1)
                pred_disp = F.interpolate(pred_disp, size=(gt_disparity.size(-2), gt_disparity.size(-1)),
                                          mode='bilinear', align_corners=False) * (
                                    gt_disparity.size(-1) / pred_disp.size(-1))
                pred_disp = pred_disp.squeeze(1)

            cur_loss = self.criterion(pred_disp[mask] * 256, gt_disparity[mask] * 256) # Note: gt_disparity was divided by 256
            loss += weight * cur_loss

        return loss
