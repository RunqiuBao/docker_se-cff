import torch.nn as nn
import torch.nn.functional as F
import torch
from typing import Optional
from thop import profile

from .refinement import StereoDRNetRefinement
from . import losses

from .feature_extractor import FeatureExtractor
from .cost import CostVolumePyramid
from .aggregation import AdaptiveAggregation
from .estimation import DisparityEstimationPyramid


class StereoMatchingNetwork(nn.Module):

    def __init__(
        self,
        network_cfg,
        loss_cfg,
        is_freeze,
        **kwargs
    ):
        max_disp = network_cfg["max_disp"]
        in_channels = network_cfg["in_channels"]
        num_downsample = network_cfg["num_downsample"]
        no_mdconv = network_cfg["no_mdconv"]
        feature_similarity = network_cfg["feature_similarity"]
        num_scales = network_cfg["num_scales"]
        num_fusions = network_cfg["num_fusions"]
        deformable_groups = network_cfg["deformable_groups"]
        mdconv_dilation = network_cfg["mdconv_dilation"]
        no_intermediate_supervision = network_cfg["no_intermediate_supervision"]
        num_stage_blocks = network_cfg["num_stage_blocks"]
        num_deform_blocks = network_cfg["num_deform_blocks"]
        refine_channels = network_cfg["refine_channels"]
        isInputFeature = network_cfg["isInputFeature"]
        self._config = network_cfg
        self._config["is_freeze"] = is_freeze

        super(StereoMatchingNetwork, self).__init__()

        refine_channels = in_channels if refine_channels is None else refine_channels
        self.num_downsample = num_downsample
        self.num_scales = num_scales

        # Feature extractor
        if not isInputFeature:
            self.feature_extractor = FeatureExtractor(in_channels=in_channels)
        else:
            self.feature_extractor = None
        max_disp = max_disp // 3

        # Cost volume construction
        self.cost_volume_constructor = CostVolumePyramid(max_disp, feature_similarity=feature_similarity)

        # Cost aggregation
        self.aggregation = AdaptiveAggregation(
            max_disp=max_disp,
            num_scales=num_scales,
            num_fusions=num_fusions,
            num_stage_blocks=num_stage_blocks,
            num_deform_blocks=num_deform_blocks,
            no_mdconv=no_mdconv,
            mdconv_dilation=mdconv_dilation,
            deformable_groups=deformable_groups,
            intermediate_supervision=not no_intermediate_supervision,
        )

        # Disparity estimation
        self.disparity_estimation = DisparityEstimationPyramid(max_disp)

        # Refinement
        refine_module_list = nn.ModuleList()
        for i in range(num_downsample):
            refine_module_list.append(
                StereoDRNetRefinement(img_channels=refine_channels)
            )

        self.refinement = refine_module_list

        if not self.is_freeze:
            self._disp_loss = getattr(losses, loss_cfg['NAME'])(
                loss_cfg,
                is_distributed=kwargs["is_distributed"],
                logger=kwargs["logger"]
            )

    @property
    def is_freeze(self):
        return self._config["is_freeze"]
    
    @property
    def input_shape(self):
        return (1, 1, 480, 672)

    def disparity_refinement(self, left_img, right_img, disparity):
        disparity_pyramid = []
        for i in range(self.num_downsample):
            scale_factor = 1.0 / pow(2, self.num_downsample - i - 1)

            if scale_factor == 1.0:
                curr_left_img = left_img
                curr_right_img = right_img
            else:
                curr_left_img = F.interpolate(
                    left_img,
                    scale_factor=scale_factor,
                    mode="bilinear",
                    align_corners=False,
                )
                curr_right_img = F.interpolate(
                    right_img,
                    scale_factor=scale_factor,
                    mode="bilinear",
                    align_corners=False,
                )
            inputs = (disparity, curr_left_img, curr_right_img)
            disparity = self.refinement[i](*inputs)
            disparity_pyramid.append(disparity)  # [H/2, H]

        return disparity_pyramid

    def predict(self, left_img, right_img):
        if self.feature_extractor is not None:
            left_feature = self.feature_extractor(left_img)
            right_feature = self.feature_extractor(right_img)
        else:
            left_feature, right_feature = (
                left_img,
                right_img,
            )  # Note: in this case, the inputs are already deep features.
        cost_volume = self.cost_volume_constructor(left_feature, right_feature)
        aggregation = self.aggregation(cost_volume)
        disparity_pyramid = self.disparity_estimation(aggregation)
        disparity_pyramid += self.disparity_refinement(
            left_img, right_img, disparity_pyramid[-1]
        )

        return disparity_pyramid
    
    def forward(self, left_img, right_img, labels=None, **kwargs):
        preds = self.predict(left_img, right_img)
        losses = None
        if labels is not None and not self.is_freeze:
            losses = self.compute_loss(preds, labels, left_img=left_img, right_img=right_img)
        artifacts = None
        return preds, losses, artifacts

    def compute_loss(self, preds: Optional[torch.Tensor], labels: Optional[torch.Tensor], **kwargs):
        """
        Compute loss
        """
        loss_final = self._disp_loss((
            preds,
            labels,
            kwargs["left_img"],
            kwargs["right_img"]
        ))
        return loss_final

    def get_params_group(self, learning_rate):
        def filter_specific_params(kv):
            specific_layer_name = ["offset_conv.weight", "offset_conv.bias"]
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
    def ComputeCostProfile(model):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        left_img = torch.randn(*model.input_shape).to(device)
        right_img = torch.randn(*model.input_shape).to(device)
        model = model.to(device)
        flops, numParams = profile(model, inputs=(left_img, right_img), verbose=False)
        return flops, numParams
