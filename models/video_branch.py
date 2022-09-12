"""
Some code in this file has been adapted from the following repository:
https://pytorch.org/vision/stable/_modules/torchvision/models/video/resnet.html
"""
import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Optional, Callable, List, Type, Any, Union
from torchvision.models.video.resnet import (
    Conv2Plus1D,
    BasicBlock,
    Conv3DNoTemporal,
    Conv3DSimple,
    Bottleneck,
    R2Plus1dStem,
)


PADDING_MODE = "reflect"

class VideoResnet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        conv_makers: List[Type[Union[Conv3DSimple, Conv3DNoTemporal, Conv2Plus1D]]],
        layers: List[int],
        stem: Callable[..., nn.Module],
        zero_init_residual: bool = False,
        features=[3, 64, 3],
        inject_noise=False,
        args=None,
    ) -> None:

        super(VideoResnet, self).__init__()
        inp_nf = features[0]
        hid_nf = features[1]
        out_nf = features[2]
        self.inject_noise = inject_noise
        self.inplanes = hid_nf
        self.args = args

        self.stem = stem(inp_nf, 128, hid_nf)

        self.layer1 = self._make_layer(
            block, conv_makers[0], hid_nf, layers[0], stride=1
        )
        self.downsample = nn.MaxPool3d(kernel_size=2, stride=2)
        self.layer2 = self._make_layer(
            block, conv_makers[1], hid_nf * 2, layers[1], stride=1
        )
        self.layer3 = self._make_layer(
            block, conv_makers[2], hid_nf * 2, layers[2], stride=1
        )

        if self.args.model.noise_injection == "concat":
            self.inplanes *= 2

        self.layer4 = self._make_layer(
            block, conv_makers[3], hid_nf * 2, layers[3], stride=1
        )
        self.upsample = nn.Upsample(
            scale_factor=2, mode="trilinear", align_corners=False
        )
        self.layer5 = self._make_layer(
            block, conv_makers[4], hid_nf, layers[4], stride=1
        )

        self.outlayer = nn.Conv3d(
            in_channels=hid_nf,
            out_channels=out_nf,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode=PADDING_MODE,
        )

        self._initialize_weights()

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)

        x = self.layer1(x)
        x = self.downsample(x)

        x = self.layer2(x)
        x = self.layer3(x)
        if self.inject_noise:
            if self.args.model.noise_injection == "mult":
                x = self.inject_noise(x)
            elif self.args.model.noise_injection == "add":
                raise NotImplementedError("Add noise not implemented")
            elif self.args.model.noise_injection == "concat":
                x = torch.cat([x, self.inject_noise.get_noise(x)], dim=1)
        x = self.layer4(x)

        x = self.upsample(x)
        x = self.layer5(x)

        x = self.outlayer(x)

        return x

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        conv_builder: Type[Union[Conv3DSimple, Conv3DNoTemporal, Conv2Plus1D]],
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            ds_stride = conv_builder.get_downsample_stride(stride)
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=ds_stride,
                    bias=False,
                    padding_mode=PADDING_MODE,
                ),
                nn.BatchNorm3d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, conv_builder, stride, downsample))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, conv_builder))

        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class VideoStem(nn.Sequential):
    """R(2+1)D stem is different than the default one as it uses separated 3D convolution"""

    def __init__(self, inp, mid, out, stride=1) -> None:
        super(VideoStem, self).__init__(
            nn.Conv3d(
                inp,
                mid,
                kernel_size=(1, 7, 7),
                stride=(1, stride, stride),
                padding=(0, 3, 3),
                bias=False,
                padding_mode=PADDING_MODE,
            ),
            nn.BatchNorm3d(mid),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                mid,
                out,
                kernel_size=(3, 1, 1),
                stride=(1, 1, 1),
                padding=(1, 0, 0),
                bias=False,
                padding_mode=PADDING_MODE,
            ),
            nn.BatchNorm3d(out),
            nn.ReLU(inplace=True),
        )


class VideoBranch(nn.Module):
    def __init__(self, args, noise_injector) -> None:
        super().__init__()
        self.args = args
        nf = args.model.internal_dim
        self.processXZaddUy = VideoResnet(
            block=BasicBlock,
            conv_makers=[Conv2Plus1D] * 5,
            layers=[2, 2, 2, 2, 2],
            stem=VideoStem,
            features=[self.args.model.channels + 1, nf, self.args.model.channels],
            inject_noise=noise_injector,
            args=args,
        )
        self.last_act = nn.Sigmoid()
    
    def forward(self, x, z):
        # Z is the confounder
        # X is the input ejection fraction
        # Uy is the latent noise

        # Repeat EF scalars to form a matrix matching the shape of the videos
        shaped_x = torch.ones((z.shape), device=z.device) * x[:, None, None, None, None]

        # Concatenate the confounder and the EF
        xz = torch.cat([shaped_x, z], dim=1)

        processed_xzu = self.processXZaddUy(xz)

        return self.last_act(processed_xzu)