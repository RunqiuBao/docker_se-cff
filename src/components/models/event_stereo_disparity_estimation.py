import torch.nn as nn
import torch
import numpy
from thop import profile


class EventStereoDisparityEstimationNetwork(nn.Module):
    objectaware_stereo_disparity_estimation_net = None

    def __init__(self, config_OSDENet):
        super(EventStereoDisparityEstimationNetwork, self).__init__()
        self.objectaware_stereo_disparity_estimation_net = ObjectawareStereoDisparityEstimationNet(**config_OSDENet)

    def forward(self, inputs, gt_disparity=None):
        pass
        
