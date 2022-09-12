import torch
import torch.nn as nn
import torchvision
from torch import Tensor
from torchvision.models.video.resnet import (
    Conv2Plus1D,
    BasicBlock,
    Conv3DNoTemporal,
    Conv3DSimple,
    Bottleneck,
    R2Plus1dStem,
)
from typing import Tuple, Optional, Callable, List, Type, Any, Union

from .base import BaseModel
from .video_branch import VideoStem


class EFExpert(BaseModel):
    def __init__(self, args) -> None:
        super().__init__(args)

        self.args = args

        self.model = EFExpertCore(
            block=BasicBlock,
            conv_makers=[Conv2Plus1D] * 4,
            layers=[2, 2, 2, 2],
            stem=VideoStem,
            num_classes=1,
            nf=self.args.model.internal_dim,
        )

    def forward(self, x):
        return self.model(x)  # (B, C, T, H, W) -> (B, C, T, H, W)


class EFExpertCore(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        conv_makers: List[Type[Union[Conv3DSimple, Conv3DNoTemporal, Conv2Plus1D]]],
        layers: List[int],
        stem: Callable[..., nn.Module],
        num_classes: int = 400,
        zero_init_residual: bool = False,
        nf=64,
    ) -> None:

        super(EFExpertCore, self).__init__()
        self.nf = nf
        self.inplanes = nf

        self.stem = stem(1, nf // 2, nf, stride=2)
        strides = [1, 2, 2, 2]

        self.layer0 = self._make_layer(
            block, conv_makers[0], nf * (2 ** 0), layers[0], stride=strides[0]
        )
        self.layer1 = self._make_layer(
            block, conv_makers[1], nf * (2 ** 1), layers[1], stride=strides[1]
        )
        self.layer2 = self._make_layer(
            block, conv_makers[2], nf * (2 ** 2), layers[2], stride=strides[2]
        )
        self.layer3 = self._make_layer(
            block, conv_makers[3], nf * (2 ** 3), layers[3], stride=strides[3]
        )

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(
            nf * (2 ** (len(conv_makers) - 1)) * block.expansion, num_classes
        )

        # init weights
        self._initialize_weights()

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)

        # Flatten the layer to fc
        x = x.flatten(1)
        x = self.fc(x)

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

