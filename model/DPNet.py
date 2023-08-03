import os

import torch
from torch import nn
from torchvision.models import resnet50,vgg16_bn,vision_transformer
import torch.nn.functional as F
from functools import partial

import math
from torchinfo.torchinfo import summary
from .resnet import resnet50 as SPNet
from .PVT import PyramidVisionTransformerV2
from timm.models.vision_transformer import _cfg

class DPNet(nn.Module):
    def __init__(self,classes=2,AA = True):
        super(DPNet, self).__init__()
        spnet_dim_list = [256,512,1024,2048]
        pvt_dim_list = [64,128,320,512]
        self.SPNet = SPNet()
        self.PVT = TransformerBlock()
        self.withAA = AA
        self.conv1 = nn.Sequential(nn.Conv2d(2048,128,1,1,bias=False),nn.BatchNorm2d(128),nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(512, 128, 1, 1, bias=False), nn.BatchNorm2d(128),nn.ReLU())
        self.AA = AggregateAttention(in_dim=256)

        self.spnet1 = nn.Sequential(nn.Conv2d(spnet_dim_list[0],64,1,1,bias=False),
                                    nn.GroupNorm(32,64),
                                    nn.ReLU())
        self.spnet2 = nn.Sequential(nn.Conv2d(spnet_dim_list[1], 64, 1, 1,bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU())
        self.spnet3 = nn.Sequential(nn.Conv2d(spnet_dim_list[2], 64, 1, 1,bias=False),
                                    nn.GroupNorm(32,64),
                                    nn.ReLU())
        self.spnet4 = nn.Sequential(nn.Conv2d(spnet_dim_list[3], 64, 1, 1,bias=False),
                                    nn.GroupNorm(32,64),
                                    nn.ReLU())

        self.pvtnet2 = nn.Sequential(nn.Conv2d(pvt_dim_list[1], 64, 1, 1,bias=False),
                                     nn.GroupNorm(32,64),
                                     nn.ReLU())
        self.pvtnet3 = nn.Sequential(nn.Conv2d(pvt_dim_list[2], 64, 1, 1,bias=False),
                                     nn.GroupNorm(32,64),
                                     nn.ReLU())
        self.pvtnet4 = nn.Sequential(nn.Conv2d(pvt_dim_list[3], 64, 1, 1,bias=False),
                                     nn.GroupNorm(32,64),
                                     nn.ReLU())

        self.CBBlock = Latent_Conv(128,128)


        self.head = nn.Sequential(ResBlock(128, 64), ResBlock(64, 64), nn.Conv2d(64, classes, 1))


    def Concat_Feature(self, x, y):
        out = torch.cat((x, y), dim=1)
        return out

    def get_forward(self,x,train=True):
        B,C,H,W = x.size()
        x1 = self.SPNet(x)
        s1 = self.spnet1(x1[0])
        s2 = self.spnet2(x1[1])
        s3 = self.spnet3(x1[2])
        s4 = self.spnet4(x1[3])

        x2 = self.PVT(x)
        p1 = x2[0]
        p2 = self.pvtnet2(x2[1])
        p3 = self.pvtnet3(x2[2])
        p4 = self.pvtnet4(x2[3])

        concat1 = self.Concat_Feature(s4,p4)
        if self.withAA:
            conv1 = self.conv1(x1[3])
            conv2 = self.conv2(x2[3])
            concat = self.Concat_Feature(conv1,conv2)
            AA = self.AA(concat)
            concat1 = AA + concat1
        CB1 = self.CBBlock(concat1)

        concat2 = self.Concat_Feature(s3,p3) + CB1
        CB2 = self.CBBlock(concat2)

        concat3 = self.Concat_Feature(s2,p2) + CB2
        CB3 = self.CBBlock(concat3)

        concat4 = self.Concat_Feature(s1,p1) + CB3

        upsample = F.interpolate(concat4, (H, W), mode='bilinear', align_corners=True)

        head = self.head(upsample)

        if train:
            head3 = self.head(F.interpolate(concat3, (H, W), mode='bilinear', align_corners=True))
            head2 = self.head(F.interpolate(concat2, (H, W), mode='bilinear', align_corners=True))
            head1 = self.head(F.interpolate(concat1, (H, W), mode='bilinear', align_corners=True))
            out = [head, head3, head2, head1]
            return out
        # Dict = {'SPNet0': x1[0], 'SPNet1': x1[1], 'SPNet2': x1[2], 'SPNet3': x1[3],
        #         'PVT0': x2[0], 'PVT1': x2[1], 'PVT2': x2[2], 'PVT3': x2[3],
        #         'Upsample': upsample}
        out = head
        return out

    def forward(self, x,train=True):
        out = self.get_forward(x,train)

        return out

class Latent_Conv(nn.Module):
    def __init__(self,in_dim,out_dim):
        super(Latent_Conv,self).__init__()
        self.conv1 = nn.Conv2d(in_dim,out_dim,3)
        self.BN1 = nn.GroupNorm(32,out_dim)
        self.relu = nn.ReLU()
        self.RB = ResBlock(in_dim,out_dim)
    def forward(self,x):
        _,_,h,w = x.size()
        out = self.RB(x)

        out = F.interpolate(out,scale_factor=2,mode='bilinear',align_corners=True)
        return out


class TransformerBlock(nn.Module):
    def __init__(self):

        super().__init__()

        backbone = PyramidVisionTransformerV2(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            depths=[3, 4, 18, 3],
            sr_ratios=[8, 4, 2, 1],
        )
        path = os.path.join(os.path.dirname(__file__),r'pvt_v2_b3.pth')
        checkpoint = torch.load(path)
        backbone.default_cfg = _cfg()
        backbone.load_state_dict(checkpoint)
        self.backbone = torch.nn.Sequential(*list(backbone.children()))[:-1]

        for i in [1, 4, 7, 10]:
            self.backbone[i] = torch.nn.Sequential(*list(self.backbone[i].children()))

    def get_pyramid(self, x):
        pyramid = []
        B = x.shape[0]
        for i, module in enumerate(self.backbone):
            if i in [0, 3, 6, 9]:
                x, H, W = module(x)
            elif i in [1, 4, 7, 10]:
                for sub_module in module:
                    x = sub_module(x, H, W)
            else:
                x = module(x)
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                pyramid.append(x)

        return pyramid

    def forward(self, x):
        pyramid = self.get_pyramid(x)

        return pyramid

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )

        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

        if out_channels == in_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        h = self.in_layers(x)
        h = self.out_layers(h)
        return h + self.skip(x)

class AggregateAttention(nn.Module):
    def __init__(self, in_dim):
        super(AggregateAttention, self).__init__()
        self.channel_in = in_dim

        self.q_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.k_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.v_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
        self.downdim = nn.Sequential(nn.Conv2d(in_dim,128,1,1),
                                     nn.GroupNorm(32,128),
                                     nn.ReLU())
    def forward(self, x):
        batch, c, h, w = x.size()
        proj_q = self.q_conv(x).view(batch, -1, w * h).permute(0, 2, 1)  # (1,100,32)
        proj_k = self.q_conv(x).view(batch, -1, w * h)  # (1,32,100)
        energy = torch.bmm(proj_q, proj_k)  # (1,100,100)
        attention = self.softmax(energy)  # (1,100,100)
        proj_v = self.v_conv(x).view(batch, -1, w * h)
        out = torch.bmm(proj_v, attention.permute(0, 2, 1))
        out = out.view(batch, c, h, w)

        out = self.gamma * out + x
        out = self.downdim(out)
        return out


if __name__ == '__main__':
    # for i in reversed(range(4)):
    #     print(i)
    inputdata = torch.randn((2, 3, 352, 352))
    inputdata = inputdata.cuda()
    dpnet = DPNet(2,False).cuda()
    # print(fpvt)
    out,_ = dpnet(inputdata,train=False)
    print(out.shape)
    # tb = TB().cuda()
    # TB_P = tb(inputdata)
    # print(TB_P[0].shape,TB_P[1].shape,TB_P[2].shape,TB_P[3].shape)
    summary(dpnet,(2,3,352,352))