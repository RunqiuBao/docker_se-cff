import torch.nn as nn
import torch


class Cylinder5DDetectionHead(nn.Module):
    def __init__(self, num_scales):
        pass

    def forward(self, left_input, right_input, disparity_pyramid):
        # RPN, similar to yolo, but predict oriented bbox
        
        # keypoint pred
        # stereo regression
        