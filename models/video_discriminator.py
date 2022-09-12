import torch
import torch.nn as nn
from einops import rearrange

from .base import BaseModel


class MultiScaleBlock(nn.Module):
    def __init__(self, inc, nf, downsample=True, make_3D=False) -> None:
        super().__init__()

        if make_3D:
            conv = nn.Conv3d
            bn = nn.BatchNorm3d
            mxp = nn.MaxPool3d
        else:
            conv = nn.Conv2d
            bn = nn.BatchNorm2d
            mxp = nn.MaxPool2d

        self.layer3 = nn.Sequential(
            conv(inc, nf, kernel_size=3, stride=1, padding=1, bias=False),
            bn(nf, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            mxp(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
            if downsample
            else nn.Identity(),
        )
        self.layer5 = nn.Sequential(
            conv(inc, nf, kernel_size=5, stride=1, padding=2, bias=False),
            bn(nf, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            mxp(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
            if downsample
            else nn.Identity(),
        )
        self.layer7 = nn.Sequential(
            conv(inc, nf, kernel_size=7, stride=1, padding=3, bias=False),
            bn(nf, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            mxp(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
            if downsample
            else nn.Identity(),
        )
        self.maxpool = (
            mxp(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
            if downsample
            else nn.Identity()
        )

    def forward(self, x):
        return (
            self.layer3(x) + self.layer5(x) + self.layer7(x) + self.maxpool(x)
        ) / 4.0


class VideoDiscriminator(BaseModel):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.args = args
        nf = args.model.internal_dim
        self.make_3D = args.model.discriminator_3D
        inc = self.args.model.channels

        self.net = nn.Sequential(
            MultiScaleBlock(inc, nf, make_3D=self.make_3D),  # -> 56
            MultiScaleBlock(nf, nf, make_3D=self.make_3D),  # -> 28
            MultiScaleBlock(nf, nf, make_3D=self.make_3D),  # -> 14
            MultiScaleBlock(nf, nf, make_3D=self.make_3D),  # -> 7
            MultiScaleBlock(nf, nf, downsample=False, make_3D=self.make_3D),  # -> 7
        )
        self.last_act = nn.Sigmoid()
        self.forward = self.forward_3D if self.make_3D else self.forward_2D

    def get_parameters(self):
        return self.parameters()

    def with_weights(self, path):
        loaded_dict = torch.load(path)
        self.load_state_dict(loaded_dict["d_weights"])
        return self

    def forward_2D(self, x):
        B, C, F, H, W = x.shape
        x = rearrange(x, "b c f h w -> (b f) c h w")
        x = self.net(x)
        x = rearrange(x, "(b f) c h w -> b c f h w", b=B, f=F)
        x = self.last_act(x.mean(dim=[1, 2, 3, 4]))
        return x

    def forward_3D(self, x):
        x = self.net(x)
        x = self.last_act(x.mean(dim=[1, 2, 3, 4]))
        return x
