"""Dilated ResNet"""

import math
import torch
# import torch.utils.model_zoo as model_zoo
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['ResNet', 'resnet50', 'resnet101', 'BasicBlock', 'Bottleneck','biresnet50','StripPooling']

#自定义3*3卷积模块
def conv3x3(in_planes, out_planes, stride=1):
    # "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

#残差网络基础块
class BasicBlock(nn.Module):
    # """ResNet BasicBlock
    # """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, previous_dilation=1,
                 norm_layer=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=previous_dilation, dilation=previous_dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

#SPM
class SPBlock(nn.Module):
    def __init__(self, inplanes, outplanes, norm_layer=None):
        super(SPBlock, self).__init__()
        midplanes = outplanes
        self.conv1 = nn.Conv2d(inplanes, midplanes, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.bn1 = norm_layer(midplanes)
        self.conv2 = nn.Conv2d(inplanes, midplanes, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.bn2 = norm_layer(midplanes)
        self.conv3 = nn.Conv2d(midplanes, outplanes, kernel_size=1, bias=True)
        self.pool1 = nn.AdaptiveAvgPool2d((None, 1))
        self.pool2 = nn.AdaptiveAvgPool2d((1, None))
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        _, _, h, w = x.size()
        x1 = self.pool1(x)
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = x1.expand(-1, -1, h, w)
        # x1 = F.interpolate(x1, (h, w))

        x2 = self.pool2(x)
        x2 = self.conv2(x2)
        x2 = self.bn2(x2)
        x2 = x2.expand(-1, -1, h, w)
        # x2 = F.interpolate(x2, (h, w))

        x = self.relu(x1 + x2)
        x = self.conv3(x).sigmoid()
        return x


class Bottleneck(nn.Module):
    # """ResNet Bottleneck
    # """
    # pylint: disable=unused-argument
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1,
                 downsample=None, previous_dilation=1, norm_layer=None, spm_on=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride,
            padding=dilation, dilation=dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

        self.spm = None
        if spm_on:
            self.spm = SPBlock(planes, planes, norm_layer=norm_layer)

    def _sum_each(self, x, y):
        assert (len(x) == len(y))
        z = []
        for i in range(len(x)):
            z.append(x[i] + y[i])
        return z

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        if self.spm is not None:
            out = out * self.spm(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    # """Dilated Pre-trained ResNet Model, which preduces the stride of 8 featuremaps at conv5.
    # Parameters
    # ----------
    # block : Block
    #     Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    # layers : list of int
    #     Numbers of layers in each block
    # classes : int, default 1000
    #     Number of classification classes.
    # dilated : bool, default False
    #     Applying dilation strategy to pretrained ResNet yielding a stride-8 model,
    #     typically used in Semantic Segmentation.
    # norm_layer : object
    #     Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
    #     for Synchronized Cross-GPU BachNormalization).
    # Reference:
    #     - He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
    #     - Yu, Fisher, and Vladlen Koltun. "Multi-scale context aggregation by dilated convolutions."
    # """

    # pylint: disable=unused-variable
    def __init__(self, block, layers, num_classes=1000, dilated=False, multi_grid=False,
                 deep_base=True, norm_layer=nn.BatchNorm2d, spm_on=False):
        self.inplanes = 128 if deep_base else 64
        super(ResNet, self).__init__()
        self.layers = layers

        self.spm_on = spm_on
        if deep_base:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
                norm_layer(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                norm_layer(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            )
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        if layers[2] > 0 and layers[3] > 0:
            if dilated:
                self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                                               dilation=2, norm_layer=norm_layer)
                if multi_grid:
                    self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                                   dilation=4, norm_layer=norm_layer,
                                                   multi_grid=True)
                else:
                    self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                                   dilation=4, norm_layer=norm_layer)
            else:
                self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                               norm_layer=norm_layer)
                self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                               norm_layer=norm_layer)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, norm_layer):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=None, multi_grid=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        spm_on = False
        if planes == 512:
            spm_on = self.spm_on

        layers = []
        multi_dilations = [4, 8, 16]
        if multi_grid:
            layers.append(block(self.inplanes, planes, stride, dilation=multi_dilations[0],
                                downsample=downsample, previous_dilation=dilation, norm_layer=norm_layer,
                                spm_on=spm_on))
        elif dilation == 1 or dilation == 2:
            layers.append(block(self.inplanes, planes, stride, dilation=1,
                                downsample=downsample, previous_dilation=dilation, norm_layer=norm_layer,
                                spm_on=spm_on))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, dilation=4,
                                downsample=downsample, previous_dilation=dilation, norm_layer=norm_layer,
                                spm_on=spm_on))
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i >= blocks - 1 or planes == 512:
                spm_on = self.spm_on
            if multi_grid:
                layers.append(block(self.inplanes, planes, dilation=multi_dilations[i],
                                    previous_dilation=dilation, norm_layer=norm_layer, spm_on=spm_on))
            else:
                layers.append(block(self.inplanes, planes, dilation=dilation, previous_dilation=dilation,
                                    norm_layer=norm_layer, spm_on=spm_on))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)


        if self.layers[2] != 0 and self.layers[3] != 0:
            x1 = self.layer1(x)
            x2 = self.layer2(x1)
            x3 = self.layer3(x2)
            x4 = self.layer4(x3)
            feature = [x1,x2,x3,x4]
            return feature
        else:
            x1 = self.layer1(x)
            x2 = self.layer2(x1)
            feature = [x1,x2]
            return feature

class StripPooling(nn.Module):
    """
    Reference:
    """
    def __init__(self, in_channels, pool_size, norm_layer, up_kwargs):
        super(StripPooling, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(pool_size[0])
        self.pool2 = nn.AdaptiveAvgPool2d(pool_size[1])
        self.pool3 = nn.AdaptiveAvgPool2d((1, None))
        self.pool4 = nn.AdaptiveAvgPool2d((None, 1))

        inter_channels = int(in_channels/4)
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv1_2 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv2_0 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels))
        self.conv2_1 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels))
        self.conv2_2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels))
        self.conv2_3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (1, 3), 1, (0, 1), bias=False),
                                norm_layer(inter_channels))
        self.conv2_4 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (3, 1), 1, (1, 0), bias=False),
                                norm_layer(inter_channels))
        self.conv2_5 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv2_6 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(inter_channels*2, in_channels, 1, bias=False),
                                norm_layer(in_channels))
        # bilinear interpolate options
        self._up_kwargs = up_kwargs

    def forward(self, x):
        _, _, h, w = x.size()
        x1 = self.conv1_1(x)
        x2 = self.conv1_2(x)
        x2_1 = self.conv2_0(x1)
        x2_2 = F.interpolate(self.conv2_1(self.pool1(x1)), (h, w), mode=self._up_kwargs,align_corners=True)
        x2_3 = F.interpolate(self.conv2_2(self.pool2(x1)), (h, w), mode=self._up_kwargs,align_corners=True)
        x2_4 = F.interpolate(self.conv2_3(self.pool3(x2)), (h, w), mode=self._up_kwargs,align_corners=True)
        x2_5 = F.interpolate(self.conv2_4(self.pool4(x2)), (h, w), mode=self._up_kwargs,align_corners=True)
        x1 = self.conv2_5(F.relu_(x2_1 + x2_2 + x2_3))
        x2 = self.conv2_6(F.relu_(x2_5 + x2_4))
        out = self.conv3(torch.cat([x1, x2], dim=1))
        return F.relu_(x + out)

def biresnet50(pretrained=None,**kwargs):
    model = ResNet(Bottleneck,[3,4,0,0],dilated=True,spm_on=True)
    return model


def resnet50(pretrained=None, **kwargs):
    # """Constructs a ResNet-50 model.
    # Args:
    #     pretrained (bool): If True, returns a model pre-trained on ImageNet
    # """
    model = ResNet(Bottleneck, [3, 4, 6, 3],dilated=False,spm_on=True)
    if pretrained:
        model.load_state_dict(torch.load(pretrained), strict=False)
    return model


def resnet101(pretrained=None, **kwargs):
    # """Constructs a ResNet-101 model.
    # Args:
    #     pretrained (bool): If True, returns a model pre-trained on ImageNet
    # """
    model = ResNet(Bottleneck, [3, 4, 23, 3],dilated=True,spm_on=True)
    if pretrained:
        model.load_state_dict(torch.load(pretrained), strict=False)
    return model

if __name__ == '__main__':
    inputdata = torch.randn((2,3,352,352))
    inputdata = inputdata.cuda()
    resnet = resnet50().cuda()
    out = resnet(inputdata)
    print(out[0].shape,out[1].shape,out[2].shape,out[3].shape)
    # print('\n\n\n................\n\n\n')
    # print(resnet50())