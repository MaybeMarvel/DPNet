import torch
from torch import nn
from torchvision.models import resnet50,vgg16_bn,vision_transformer
import torch.nn.functional as F
from functools import partial

import math
from torchinfo.torchinfo import summary
from projectcode.model.resnet import resnet50 as r50
from projectcode.model.resnet import biresnet50,StripPooling
from projectcode.model import PVT
from timm.models.vision_transformer import _cfg


class FPN(nn.Module):
    def __init__(self,num_classes=32):
        super(FPN, self).__init__()
        self.num_classes = num_classes

        #bottom-to-top
        self.backbone = resnet50(pretrained=True)
        self.resnet50_head = nn.Sequential(*list(resnet50().children())[:4])#3->64
        self.layer1 = self.backbone.layer1
        self.layer2 = self.backbone.layer2
        self.layer3 = self.backbone.layer3
        self.layer4 = self.backbone.layer4

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
        c1 = self.resnet50_head(inputs)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        # CAM1 = self.CAM(self.latenlayer1(c4))
        # print(CAM1.shape)

        #Top-down
        p5 = self.attention(self.toplayer(c5))
        # print(p5.shape,self.latenlayer1(c4).shape)
        p4 = self.upsample(p5,self.CAM(self.latenlayer1(c4)))
        p3 = self.upsample(p4,self.CAM(self.latenlayer2(c3)))
        p2 = self.upsample(p3,self.CAM(self.latenlayer3(c2)))


        #Smooth
        p4 = self.smooth1(self.attention(p4))
        p3 = self.smooth2(self.attention(p3))
        p2 = self.smooth3(self.attention(p2))

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


#self-attention = position-attention
class Self_attention(nn.Module):
    def __init__(self,in_dim):
        super(Self_attention,self).__init__()
        self.channel_in = in_dim

        self.q_conv = nn.Conv2d(in_channels=in_dim,out_channels=in_dim//8,kernel_size=1)
        self.k_conv = nn.Conv2d(in_channels=in_dim,out_channels=in_dim//8,kernel_size=1)
        self.v_conv = nn.Conv2d(in_channels=in_dim,out_channels=in_dim,kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self,x):
        batch,c,h,w = x.size()
        proj_q = self.q_conv(x).view(batch,-1,w*h).permute(0,2,1)#(1,100,32)
        proj_k = self.q_conv(x).view(batch,-1,w*h)#(1,32,100)
        energy = torch.bmm(proj_q,proj_k)#(1,100,100)
        attention = self.softmax(energy)#(1,100,100)
        proj_v = self.v_conv(x).view(batch,-1,w*h)
        out = torch.bmm(proj_v,attention.permute(0,2,1))
        out = out.view(batch,c,h,w)

        out = self.gamma * out + x
        return out

class CAM_Module(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out



class depthwise_separable_conv(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1,
                                   groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out



    def forward(self, Detail_x, Semantic_x):
        DetailBranch_1 = self.Conv_DetailBranch_1(Detail_x)
        DetailBranch_2 = self.Conv_DetailBranch_2(Detail_x)

        SemanticBranch_1 = self.Conv_SemanticBranch_1(Semantic_x)
        SemanticBranch_2 = self.Conv_SemanticBranch_2(Semantic_x)

        out_1 = torch.matmul(DetailBranch_1, SemanticBranch_1)
        out_2 = torch.matmul(DetailBranch_2, SemanticBranch_2)
        out_2 = F.interpolate(out_2, scale_factor=4, mode="bilinear", align_corners=True)

        out = torch.matmul(out_1, out_2)
        out = self.conv_out(out)
        return out


class SegHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(SegHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

        self.sa = Self_attention(inter_channels)
        self.sc = CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))
        self.conv7 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)
        sa_output = self.conv6(sa_conv)

        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)
        sc_output = self.conv7(sc_conv)

        feat_sum = sa_conv + sc_conv

        sasc_output = self.conv8(feat_sum)

        output = sasc_output
        return output


class FPNV1(nn.Module):
    def __init__(self,classes=32):
        super(FPNV1,self).__init__()
        self.classes = classes
        self.detail_channel = biresnet50()

        self.semantic_channel = r50()

        # self.SAM = Self_attention(256)
        # self.CAM = CAM_Module(256)

        self.Seghead = SegHead(2048,classes,norm_layer=nn.BatchNorm2d)
        self.DetailHead = nn.Sequential(nn.Conv2d(512,256,3,1,1),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(),
                                        nn.Conv2d(256,128,3,1,1),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(),
                                        nn.Conv2d(128,classes,1,1,1))
        self.JAM = JumpAttention(classes,classes)
        self.Head = nn.Sequential(nn.Dropout2d(0.1,False),
            nn.Conv2d(128,classes,1,1,0))

    def forward(self,x):
        _,_,h,w = x.size()
        detail_channel = self.detail_channel(x)
        # smooth2 = self.smooth2(detail_channel)
        DetailHead = self.DetailHead(detail_channel[1])


        semantic_channel = self.semantic_channel(x)
        # print(semantic_channel.shape)
        SegHead = self.Seghead(semantic_channel[3])
        JAM = self.JAM(DetailHead,SegHead)
        # _,_,h1,w1 = DetailHead.size()
        # semantic_out = F.interpolate(input=SegHead,size=(h1,w1),mode='bilinear',align_corners=True)
        #
        # add = semantic_out + DetailHead
        # Head = self.Head(JAM)
        out = F.interpolate(input=JAM,size=(h,w),mode='bilinear', align_corners=True)
        feature_Dict = {'detail_channel':detail_channel,'DetailHead':DetailHead,
                'semantic_channel':semantic_channel,'SegHead':SegHead,'JAM':JAM
                 }

        return out,feature_Dict

class JumpAttention(nn.Module):
    def __init__(self,inplanes,outplanes):
        super(JumpAttention,self).__init__()
        self.in_a = nn.Sequential(
            nn.Conv2d(inplanes,outplanes,1),
            nn.BatchNorm2d(outplanes),
            nn.ReLU()
        )
        self.in_b = nn.Sequential(
            nn.Conv2d(inplanes,outplanes,1),
            nn.BatchNorm2d(outplanes),
            nn.ReLU()
        )
        self.in_c = nn.Sequential(
            nn.Conv2d(inplanes,outplanes,1),
            nn.BatchNorm2d(outplanes),
            nn.ReLU()
        )
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))
    def forward(self,x,y):
        batch,c,h,w = x.size()
        # batch1,c1,h1,w1 = y.size()
        the_y = F.interpolate(y,(h,w),mode='bilinear',align_corners=True)
        in_a = self.in_a(x).view(batch,c,h*w).permute(0,2,1)
        in_b = self.in_b(the_y).view(batch,c,h*w)
        energy = torch.bmm(in_a,in_b)
        attention = self.softmax(energy)
        add = self.in_c(x +the_y).view(batch,c,h*w)
        out = torch.bmm(add,attention.permute(0,2,1))
        out = out.view(batch,c,h,w)
        out = self.gamma * out + x + the_y

        return out


if __name__ == '__main__':
    # img,mask = next(iter(train_loader))
    # gpu_img,gpu_mask = img.cuda(),mask.cuda()
    # fpn = FPN(32).cuda()
    # output = fpn(gpu_img)
    # plt.imshow(img[0].permute(1,2,0))
    # plt.show()
    # print(resnet50)
    inputdata = torch.randn((2,3,320,320))
    inputdata = inputdata.cuda()
    FPN1 = FPNV1().cuda()
    # print(FPN1)

    # out,Dict = FPN1(inputdata)
    # summary(FPN1,(8,3,352,352))

    # print(out.shape)
    # print(summary(fpn,input_size=(3,320,320)))


