import torch
import torch.nn as nn
import torch.nn.functional as F

from thop import profile


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, bias=False):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            *[
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=bias,
                ),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(negative_slope=0.1),
            ]
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(UpBlock, self).__init__()
        self.conv1 = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.conv2 = ConvBlock(
            in_channels=2 * out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.conv3 = ConvBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, x, other):
        x = F.interpolate(x, scale_factor=(2, 2))
        x = self.conv1(x)
        x = self.conv2(torch.cat([x, other], 1))
        x = self.conv3(x)

        return x


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(DownBlock, self).__init__()
        self.conv1 = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.conv2 = ConvBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.conv3 = ConvBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, x):
        x = F.avg_pool2d(x, (2, 2))
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        return x


class ConcentrationNet(nn.Module):
    def __init__(self, network_cfg, loss_cfg, is_freeze, **kwargs):
        self._config = network_cfg
        self._config["is_freeze"] = is_freeze
        in_channels, base_channels, attention_method = network_cfg["in_channels"], network_cfg["base_channels"], network_cfg["attention_method"]
        super(ConcentrationNet, self).__init__()
        self.attention_method = attention_method
        self.conv1 = ConvBlock(
            in_channels=in_channels,
            out_channels=base_channels,
            kernel_size=(3, 3),
            padding=(1, 1),
        )
        self.conv2 = ConvBlock(
            in_channels=base_channels,
            out_channels=base_channels,
            kernel_size=(3, 3),
            padding=(1, 1),
        )
        self.down1 = DownBlock(base_channels, base_channels * 2, (3, 3), (1, 1))
        self.down2 = DownBlock(base_channels * 2, base_channels * 4, (3, 3), (1, 1))
        self.down3 = DownBlock(base_channels * 4, base_channels * 8, (3, 3), (1, 1))
        self.up1 = UpBlock(base_channels * 8, base_channels * 4, (3, 3), (1, 1))
        self.up2 = UpBlock(base_channels * 4, base_channels * 2, (3, 3), (1, 1))
        self.up3 = UpBlock(base_channels * 2, base_channels, (3, 3), (1, 1))
        self.last_conv = nn.Conv2d(
            in_channels=base_channels,
            out_channels=in_channels,
            kernel_size=(3, 3),
            padding=(1, 1),
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(
                    m.weight, a=0.1, mode="fan_out", nonlinearity="leaky_relu"
                )
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @property
    def is_freeze(self):
        return self._config["is_freeze"]
    
    @property
    def input_shape(self):
        return (1, 10, 480, 672)

    def predict(self, x):
        out = self.conv1(x)
        s1 = self.conv2(out)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        out = self.down3(s3)
        out = self.up1(out, s3)
        out = self.up2(out, s2)
        out = self.up3(out, s1)
        out = self.last_conv(out)

        if self.attention_method == "hard":
            hard_attention = out.max(dim=1)[1]

            new_x = x[
                torch.arange(x.size(0), device="cuda").view(x.size(0), 1, 1, 1),
                torch.stack([hard_attention] * x.size(1), dim=1),
                torch.arange(x.size(2), device="cuda").view(1, 1, x.size(2), 1),
                torch.arange(x.size(3), device="cuda").view(1, 1, 1, x.size(3)),
            ]
            new_x = new_x.squeeze(dim=4).contiguous()
        elif self.attention_method == "soft":
            soft_attention = F.softmax(out, dim=1)
            new_x = x * soft_attention
            new_x = new_x.sum(dim=1, keepdim=True).contiguous()
        else:
            raise NotImplementedError

        return new_x
    
    def forward(self, left_img, right_img, **kwargs):
        left_preds = self.predict(left_img)
        right_preds = self.predict(right_img)
        losses = None
        artifacts = None
        return (left_preds, right_preds), losses, artifacts
    
    @staticmethod
    def ComputeCostProfile(model):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        left_img = torch.randn(*model.input_shape).to(device)
        right_img = torch.randn(*model.input_shape).to(device)
        model = model.to(device)
        flops, numParams = profile(model, inputs=(left_img, right_img), verbose=False)
        return flops, numParams
