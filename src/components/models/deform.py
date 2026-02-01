import torch.nn as nn
import torch

# from .deform_conv import DeformConv, ModulatedDeformConv
from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2d
import ctypes; import torch; import os
torch.ops.load_library("/root/code/tool_exportpytorchmodels/dist/lib/libmmdeploy_torch_ops.so")
ctypes.CDLL(os.path.join("/root/code/tool_exportpytorchmodels/dist/lib/libmmdeploy_trt_ops.so"))
if hasattr(torch.ops, 'mmdeploy'):
    print("Success: Registered Ops:", [op for op in dir(torch.ops.mmdeploy) if not op.startswith('_')])
# Define a replacement forward function that uses the registered op
def patched_forward(self, x: torch.Tensor, offset: torch.Tensor, mask: torch.Tensor):
    # Handle the case where bias is None (common in some backbones)
    bias_tensor = self.bias
    if bias_tensor is None:
        # Create a dummy empty tensor. 
        # The C++ kernel won't read it because we pass with_bias=False,
        # but the Schema Parser demands a valid Tensor object here.
        bias_tensor = torch.empty(0, device=x.device, dtype=x.dtype)
    # This matches the schema you defined in bind.cpp for modulated_deform_conv
    return torch.ops.mmdeploy.modulated_deform_conv(
        x, 
        self.weight, 
        bias_tensor,   # <--- NOW ALWAYS A TENSOR (Never None)
        offset, 
        mask,
        self.kernel_size[0], self.kernel_size[1],
        self.stride[0], self.stride[1],
        self.padding[0], self.padding[1],
        self.dilation[0], self.dilation[1],
        self.groups, 
        self.deform_groups, 
        self.bias is not None # Flag tells C++ whether to use the tensor
    )
@torch.library.register_fake("mmdeploy::modulated_deform_conv")
def _modulated_deform_conv_fake(input, weight, bias, offset, mask, 
                                kernel_h, kernel_w, stride_h, stride_w, 
                                pad_h, pad_w, dilation_h, dilation_w, 
                                groups, deform_groups, with_bias):
    
    # 1. Get input dimensions
    batch_size, _, in_h, in_w = input.shape
    out_channels = weight.shape[0] # Weight is (C_out, C_in/g, kH, kW)

    # 2. Calculate Output Height
    # Formula: floor((H + 2*padding - dilation*(kernel-1) - 1)/stride + 1)
    out_h = (in_h + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    
    # 3. Calculate Output Width
    out_w = (in_w + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1

    # 4. Return an empty tensor on the 'meta' device with the correct shape and dtype
    return input.new_empty((batch_size, out_channels, out_h, out_w))
ModulatedDeformConv2d.forward = patched_forward


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class DeformConv2d(nn.Module):
    """A single (modulated) deformable conv layer"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dilation=2,
        groups=1,
        deformable_groups=2,
        modulation=True,
        double_mask=True,
        bias=False,
    ):
        super(DeformConv2d, self).__init__()

        self.modulation = modulation
        self.deformable_groups = deformable_groups
        self.kernel_size = kernel_size
        self.double_mask = double_mask

        # if self.modulation:
            # self.deform_conv = ModulatedDeformConv(
                # in_channels,
                # out_channels,
                # kernel_size=kernel_size,
                # stride=stride,
                # padding=dilation,
                # dilation=dilation,
                # groups=groups,
                # deformable_groups=deformable_groups,
                # bias=bias,
            # )
        # else:
            # self.deform_conv = DeformConv(
                # in_channels,
                # out_channels,
                # kernel_size=kernel_size,
                # stride=stride,
                # padding=dilation,
                # dilation=dilation,
                # groups=groups,
                # deformable_groups=deformable_groups,
                # bias=bias,
            # )
        self.deform_conv = ModulatedDeformConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            groups=groups,
            deform_groups=deformable_groups,
            bias=bias,
        )

        k = 3 if self.modulation else 2

        offset_out_channels = deformable_groups * k * kernel_size * kernel_size

        # Group-wise offset leraning when deformable_groups > 1
        self.offset_conv = nn.Conv2d(
            in_channels,
            offset_out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            groups=deformable_groups,
            bias=True,
        )

        # Initialize the weight for offset_conv as 0 to act like regular conv
        nn.init.constant_(self.offset_conv.weight, 0.0)
        nn.init.constant_(self.offset_conv.bias, 0.0)

    def forward(self, x):
        if self.modulation:
            offset_mask = self.offset_conv(x)

            offset_channel = (
                self.deformable_groups * 2 * self.kernel_size * self.kernel_size
            )
            offset = offset_mask[:, :offset_channel, :, :]

            mask = offset_mask[:, offset_channel:, :, :]
            mask = mask.sigmoid()  # [0, 1]

            if self.double_mask:
                mask = mask * 2  # initialize as 1 to work as regular conv

            out = self.deform_conv(x, offset, mask)

        else:
            offset = self.offset_conv(x)
            out = self.deform_conv(x, offset)

        return out


class DeformBottleneck(nn.Module):
    expansion = 4
    __constants__ = ["downsample"]

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(DeformBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = DeformConv2d(width, width, stride=stride)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class SimpleBottleneck(nn.Module):
    """Simple bottleneck block without channel expansion"""

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(SimpleBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes)
        self.bn3 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class DeformSimpleBottleneck(nn.Module):
    """Used for cost aggregation"""

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        norm_layer=None,
        mdconv_dilation=2,
        deformable_groups=2,
        modulation=True,
        double_mask=True,
    ):
        super(DeformSimpleBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = DeformConv2d(
            width,
            width,
            stride=stride,
            dilation=mdconv_dilation,
            deformable_groups=deformable_groups,
            modulation=modulation,
            double_mask=double_mask,
        )
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes)
        self.bn3 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
