import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
import os
import math
from torchinfo.torchinfo import summary
from .resnet import resnet50 as r50
from .PVT import PyramidVisionTransformerV2
from timm.models.vision_transformer import _cfg


class SPResNet(nn.Module):
    def __init__(self,num_classes=2):
        super(SPResNet, self).__init__()
        self.num_classes = num_classes

        #bottom-to-top
        self.backbone = r50()


        #latentlayer -> 256dim
        self.toplayer = nn.Conv2d(2048,256,kernel_size=(1,1),stride=(1,1),padding=0)
        self.latenlayer1 = nn.Conv2d(1024,256,kernel_size=(1,1),stride=(1,1),padding=0)
        self.latenlayer2 = nn.Conv2d(512,256,kernel_size=(1,1),stride=(1,1),padding=0)
        self.latenlayer3 = nn.Conv2d(256,256,kernel_size=(1,1),stride=(1,1),padding=0)

        #Smooth-layer
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        #extract
        self.extract = nn.Conv2d(256,256,kernel_size=(3,3),stride=(1,1),padding=1)

        # Semantic branch
        self.semantic_branch = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, self.num_classes, kernel_size=1, stride=1, padding=0)
        # num_groups, num_channels
        self.gn1 = nn.GroupNorm(128, 128)
        self.gn2 = nn.GroupNorm(256, 256)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        # print(self.layer1)
        # print(self.resnet50_head)

    def _upsample(self, x, h, w):
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

    def upsample(self,x,y):
        _,_,h,w = y.size()
        return F.interpolate(x,size=(h,w),mode='bilinear') + y

    def forward(self,inputs):
        # attn = []
        #Bottom-up
        backlist = self.backbone(inputs)
        c2 = backlist[0]
        c3 = backlist[1]
        c4 = backlist[2]
        c5 = backlist[3]
        # CAM1 = self.CAM(self.latenlayer1(c4))
        # print(CAM1.shape)

        #Top-down
        p5 = self.toplayer(c5)
        # print(p5.shape,self.latenlayer1(c4).shape)
        p4 = self.upsample(p5,self.latenlayer1(c4))
        p3 = self.upsample(p4,self.latenlayer2(c3))
        p2 = self.upsample(p3,self.latenlayer3(c2))


        #Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)

        #Semantic
        _,_,h,w = p2.size()
        s5 = self._upsample(F.relu(self.gn2(self.conv2(p5))),h,w)
        s5 = self._upsample(F.relu(self.gn2(self.conv2(s5))), h, w)
        s5 = self._upsample(F.relu(self.gn1(self.semantic_branch(s5))), h, w)

        s4 = self._upsample(F.relu(self.gn2(self.conv2(p4))), h, w)
        s4 = self._upsample(F.relu(self.gn1(self.semantic_branch(s4))), h, w)

        s3 = self._upsample(F.relu(self.gn1(self.semantic_branch(p3))), h, w)

        s2 = F.relu(self.gn1(self.semantic_branch(p2)))

        output = self._upsample(self.conv3(s2 + s3 + s4 + s5), 4 * h, 4 * w)

        return output

class PVT_single(nn.Module):
    def __init__(self,num_class=2):
        super().__init__()
        self.num_classes = num_class
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
        path = os.path.join(os.path.dirname(__file__), r'pvt_v2_b3.pth')
        checkpoint = torch.load(path)
        backbone.default_cfg = _cfg()
        backbone.load_state_dict(checkpoint)
        self.backbone = torch.nn.Sequential(*list(backbone.children()))[:-1]

        for i in [1, 4, 7, 10]:
            self.backbone[i] = torch.nn.Sequential(*list(self.backbone[i].children()))

        # latentlayer -> 256dim
        self.toplayer = nn.Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.latenlayer1 = nn.Conv2d(320, 64, kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.latenlayer2 = nn.Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.latenlayer3 = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), padding=0)

        # Smooth-layer
        self.smooth1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # extract
        self.extract = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=1)

        # Semantic branch
        self.semantic_branch = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, self.num_classes, kernel_size=1, stride=1, padding=0)
        # num_groups, num_channels
        self.gn1 = nn.GroupNorm(32, 32)
        self.gn2 = nn.GroupNorm(64, 64)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _upsample(self, x, h, w):
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

    def upsample(self,x,y):
        _,_,h,w = y.size()
        return F.interpolate(x,size=(h,w),mode='bilinear',align_corners=True) + y

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
        c2 = pyramid[0]
        c3 = pyramid[1]
        c4 = pyramid[2]
        c5 = pyramid[3]

        # Top-down
        p5 = self.toplayer(c5)
        # print(p5.shape,self.latenlayer1(c4).shape)
        p4 = self.upsample(p5, self.latenlayer1(c4))
        p3 = self.upsample(p4, self.latenlayer2(c3))
        p2 = self.upsample(p3, self.latenlayer3(c2))

        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)

        # Semantic
        _, _, h, w = p2.size()
        s5 = self._upsample(F.relu(self.gn2(self.conv2(p5))), h, w)
        s5 = self._upsample(F.relu(self.gn2(self.conv2(s5))), h, w)
        s5 = self._upsample(F.relu(self.gn1(self.semantic_branch(s5))), h, w)

        s4 = self._upsample(F.relu(self.gn2(self.conv2(p4))), h, w)
        s4 = self._upsample(F.relu(self.gn1(self.semantic_branch(s4))), h, w)

        s3 = self._upsample(F.relu(self.gn1(self.semantic_branch(p3))), h, w)

        s2 = F.relu(self.gn1(self.semantic_branch(p2)))

        output = self._upsample(self.conv3(s2 + s3 + s4 + s5), 4 * h, 4 * w)


        return output

model_list = {'SPResNet':SPResNet(),
              'PVT_single':PVT_single()}

if __name__ == '__main__':
    inputdata = torch.randn((2,3,352,352))
    inputdata = inputdata.cuda()
    # spresnet = SPResNet().cuda()
    # out,Dict = spresnet(inputdata)
    # summary(spresnet,(8,3,352,352))
    # print(out.shape)
    # pvtsingle = PVT_single().cuda()
    # out,Dict = pvtsingle(inputdata)
    # summary(pvtsingle,(8,3,352,352))
    # print(out.shape)
