# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import numpy as np
import paddle
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import Conv2D, BatchNorm, Linear, Dropout
from paddle.nn import AdaptiveAvgPool2D, MaxPool2D, AvgPool2D

from paddleseg import utils
from paddleseg.cvlibs import manager

MODEL_URLS = {
    "MobileNetV2_x0_25":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV2_x0_25_pretrained.pdparams",
    "MobileNetV2_x0_5":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV2_x0_5_pretrained.pdparams",
    "MobileNetV2_x0_75":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV2_x0_75_pretrained.pdparams",
    "MobileNetV2":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV2_pretrained.pdparams",
    "MobileNetV2_x1_5":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV2_x1_5_pretrained.pdparams",
    "MobileNetV2_x2_0":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV2_x2_0_pretrained.pdparams"
}

__all__ = ["MobileNetV2"]


class ConvBNLayer(nn.Layer):
    def __init__(self,
                 num_channels,
                 filter_size,
                 num_filters,
                 stride,
                 padding,
                 channels=None,
                 num_groups=1,
                 name=None,
                 use_cudnn=True):
        super(ConvBNLayer, self).__init__()

        self._conv = Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            weight_attr=ParamAttr(name=f"{name}_weights"),
            bias_attr=False,
        )

        self._batch_norm = BatchNorm(
            num_filters,
            param_attr=ParamAttr(name=f"{name}_bn_scale"),
            bias_attr=ParamAttr(name=f"{name}_bn_offset"),
            moving_mean_name=f"{name}_bn_mean",
            moving_variance_name=f"{name}_bn_variance",
        )

    def forward(self, inputs, if_act=True):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        if if_act:
            y = F.relu6(y)
        return y


class InvertedResidualUnit(nn.Layer):
    def __init__(self, num_channels, num_in_filter, num_filters, stride,
                 filter_size, padding, expansion_factor, name):
        super(InvertedResidualUnit, self).__init__()
        num_expfilter = int(round(num_in_filter * expansion_factor))
        self._expand_conv = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_expfilter,
            filter_size=1,
            stride=1,
            padding=0,
            num_groups=1,
            name=f"{name}_expand",
        )

        self._bottleneck_conv = ConvBNLayer(
            num_channels=num_expfilter,
            num_filters=num_expfilter,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            num_groups=num_expfilter,
            use_cudnn=False,
            name=f"{name}_dwise",
        )

        self._linear_conv = ConvBNLayer(
            num_channels=num_expfilter,
            num_filters=num_filters,
            filter_size=1,
            stride=1,
            padding=0,
            num_groups=1,
            name=f"{name}_linear",
        )

    def forward(self, inputs, ifshortcut):
        y = self._expand_conv(inputs, if_act=True)
        y = self._bottleneck_conv(y, if_act=True)
        y = self._linear_conv(y, if_act=False)
        if ifshortcut:
            y = paddle.add(inputs, y)
        return y


class InvresiBlocks(nn.Layer):
    def __init__(self, in_c, t, c, n, s, name):
        super(InvresiBlocks, self).__init__()

        self._first_block = InvertedResidualUnit(
            num_channels=in_c,
            num_in_filter=in_c,
            num_filters=c,
            stride=s,
            filter_size=3,
            padding=1,
            expansion_factor=t,
            name=f"{name}_1",
        )

        self._block_list = []
        for i in range(1, n):
            block = self.add_sublayer(
                f"{name}_{str(i + 1)}",
                sublayer=InvertedResidualUnit(
                    num_channels=c,
                    num_in_filter=c,
                    num_filters=c,
                    stride=1,
                    filter_size=3,
                    padding=1,
                    expansion_factor=t,
                    name=f"{name}_{str(i + 1)}",
                ),
            )
            self._block_list.append(block)

    def forward(self, inputs):
        y = self._first_block(inputs, ifshortcut=False)
        for block in self._block_list:
            y = block(y, ifshortcut=True)
        return y


@manager.BACKBONES.add_component
class MobileNet(nn.Layer):
    def __init__(self,
                 input_channels=3,
                 scale=1.0,
                 pretrained=None,
                 prefix_name=""):
        super(MobileNet, self).__init__()
        self.scale = scale

        bottleneck_params_list = [
            (1, 16, 1, 1),
            (6, 24, 2, 2),
            (6, 32, 3, 2),
            (6, 64, 4, 2),
            (6, 96, 3, 1),
            (6, 160, 3, 2),
            (6, 320, 1, 1),
        ]

        self.conv1 = ConvBNLayer(
            num_channels=input_channels,
            num_filters=int(32 * scale),
            filter_size=3,
            stride=2,
            padding=1,
            name=f"{prefix_name}conv1_1",
        )

        self.block_list = []
        in_c = int(32 * scale)
        for i, layer_setting in enumerate(bottleneck_params_list, start=2):
            t, c, n, s = layer_setting
            block = self.add_sublayer(
                f"{prefix_name}conv{i}",
                sublayer=InvresiBlocks(
                    in_c=in_c,
                    t=t,
                    c=int(c * scale),
                    n=n,
                    s=s,
                    name=f"{prefix_name}conv{i}",
                ),
            )
            self.block_list.append(block)
            in_c = int(c * scale)

        self.out_c = int(1280 * scale) if scale > 1.0 else 1280
        self.conv9 = ConvBNLayer(
            num_channels=in_c,
            num_filters=self.out_c,
            filter_size=1,
            stride=1,
            padding=0,
            name=f"{prefix_name}conv9",
        )

        self.feat_channels = [int(i * scale) for i in [16, 24, 32, 96, 1280]]
        self.pretrained = pretrained
        self.init_weight()

    def forward(self, inputs):
        feat_list = []
        y = self.conv1(inputs, if_act=True)

        for block_index, block in enumerate(self.block_list):
            y = block(y)
            if block_index in [0, 1, 2, 4]:
                feat_list.append(y)
        y = self.conv9(y, if_act=True)
        feat_list.append(y)
        return feat_list

    def init_weight(self):
        utils.load_pretrained_model(self, self.pretrained)


@manager.BACKBONES.add_component
def MobileNetV2(**kwargs):
    return MobileNet(scale=1.0, **kwargs)
