"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F
import importlib

import random 
import numpy as np 
from typing import List 
from thop import profile

from .workspace import register
from .presnet import PResNet
from .hybrid_encoder import HybridEncoder
from .rtdetrv2_decoder import  RTDETRTransformerv2
from .matcher import HungarianMatcher
from .rtdetr_criterion import RTDETRCriterion


__all__ = ['RTDETR', ]


@register()
class RTDETR(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, \
        network_cfg,
        loss_cfg,
        is_freeze,
        **kwargs
    ):
        super().__init__()
        self._config = network_cfg
        self._config["is_freeze"] = is_freeze
            
        self._backbone = PResNet(**network_cfg['backbone'].params)
        self._decoder = RTDETRTransformerv2(**network_cfg['decoder'].params)
        self._encoder = HybridEncoder(**network_cfg['encoder'].params)

        matcher = HungarianMatcher(**loss_cfg.matcher_cfg.params)
        self.loss_functor = RTDETRCriterion(matcher, **loss_cfg)

    @property
    def is_freeze(self):
        return self._config["is_freeze"]

    @property
    def input_shape(self):
        return [(1, 10, 480, 672), (1, 10, 480, 672)]

    def predict_single(self, x, targets=None):
        x = self._backbone(x)
        x = self._encoder(x)        
        x = self._decoder(x, targets)
        return x
    
    def predict(self, x, x_right, targets=None):
        x = self._backbone(x)
        x_right = self._backbone(x_right)
        x = self._encoder(x)
        x_right = self._encoder(x_right)
        x = self._decoder(x, targets)
        return x, x_right

    def forward(self, x, x_right, labels=None, **kwargs):
        x, x_right = self.predict(x, x_right, targets=labels)
        losses = None
        if labels is not None:
            losses, selected_leftdetections, corresponding_gt_labels, indices = self.compute_loss(x, labels, **kwargs)
        artifacts = [x_right]
        if labels is not None:
            artifacts.append(selected_leftdetections)
            artifacts.append(corresponding_gt_labels)
            artifacts.append(indices)
        return x, losses, artifacts

    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self 
    
    def compute_loss(self, preds: torch.Tensor, labels: torch.Tensor, **kwargs):
        if self.training:
            self.loss_functor.train()
        else:
            self.loss_functor.eval()
        (
            loss_detections,
            selected_leftdetections,
            corresponding_gt_labels,
            indices
        ) = self.loss_functor(
            preds,
            labels,
            **kwargs
        )

        return loss_detections, selected_leftdetections, corresponding_gt_labels, indices
        
    @staticmethod
    def ComputeCostProfile(model):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        input_tensor1 = torch.randn(*model.input_shape[0]).to(device)
        input_tensor2 = torch.randn(*model.input_shape[1]).to(device)
        model = model.to(device)
        flops, numParams = profile(model, inputs=(input_tensor1, input_tensor2), verbose=False)
        return flops, numParams
