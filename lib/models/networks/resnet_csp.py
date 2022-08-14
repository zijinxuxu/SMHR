# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Dequan Wang and Xingyi Zhou
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import logging
from numpy.core.fromnumeric import reshape

import torch
import torch.nn as nn
# from .DCNv2.dcn_v2 import DCN
import torch.utils.model_zoo as model_zoo
import torch.nn.init as init
import torch.nn.functional as F
from lib.models.utils import _sigmoid, _tranpose_and_gather_feat
# from lib.models.networks.resnet import ResNetBackbone
from lib.models.networks.resnet import resnet18, resnet50
from lib.models.networks.networks import ConvBlock
BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False, dilation=dilation)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :] 

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight, std=0.001)
            # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

def fill_xavier_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

def fill_kaiming_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            torch.nn.init.kaiming_normal_(m.weight.data, mode='fan_out')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

def freeze_early_layers(layers):
    for layer in layers:
        for param in layer.parameters():
            # free conv layers not batchnorm layers
            if param.dim() != 1:
                param.requires_grad = False

class EncodeUV(nn.Module):
    def __init__(self, backbone):
        super(EncodeUV, self).__init__()
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, x):
        x0 = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x0)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return x0, x4, x3, x2, x1

# class EncodeMesh(nn.Module):
#     def __init__(self, backbone, in_channel):
#         super(EncodeMesh, self).__init__()
#         self.reduce = nn.Sequential(ConvBlock(in_channel, in_channel, relu=True, norm='bn'),
#                                     ConvBlock(in_channel, 128, relu=True, norm='bn'),
#                                     ConvBlock(128, 64, kernel_size=1, padding=0, relu=False, norm='bn'))
#         self.maxpool = backbone.maxpool
#         self.layer1 = backbone.layer1
#         self.layer2 = backbone.layer2
#         self.layer3 = backbone.layer3
#         self.layer4 = backbone.layer4
#         self.avgpool = backbone.avgpool
#         self.fc = backbone.fc

#     def forward(self, x):
#         x = self.reduce(x)
#         x = self.maxpool(x)
#         x1 = self.layer1(x)
#         x2 = self.layer2(x1)
#         x3 = self.layer3(x2)
#         x4 = self.layer4(x3)
#         x = self.avgpool(x4)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)
#         return x, x4, x3, x2, x1

class PoseResNet(nn.Module):

    def __init__(self, block, layers, heads, head_conv, opt=None):
        self.opt = opt
        self.inplanes = 64
        self.heads = heads
        self.deconv_with_bias = False
        self.uv_channel = 21

        super(PoseResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        # try to add heatmap channel 21
        batch_size = opt.batch_size // len(opt.gpus)
        if 'params' in self.opt.heads:
            out_feature_size = self.opt.input_res // 8 
            init_pose_param = torch.zeros((batch_size, self.opt.heads['params'],out_feature_size,out_feature_size))
            self.register_buffer('mean_theta', init_pose_param.float())
        if opt.iterations:
            self.iterations = 3
        else:
            self.iterations = 1

        # CMR like backbone
        self.relation = [[4, 8], [4, 12], [4, 16], [4, 20], [8, 12], [8, 16], [8, 20], [12, 16], [12, 20], [16, 20], [1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]
        backbone, self.latent_size = self.get_backbone("resnet18")
        self.backbone = EncodeUV(backbone)
        # backbone2, _ = self.get_backbone("resnet50")
        # self.backbone_mesh = EncodeMesh(backbone2, 64 + self.uv_channel + len(self.relation))
        self.uv_delayer = nn.ModuleList([ConvBlock(self.latent_size[2] + self.latent_size[1], self.latent_size[2], kernel_size=3, relu=True, norm='bn'),
                                         ConvBlock(self.latent_size[3] + self.latent_size[2], self.latent_size[3], kernel_size=3, relu=True, norm='bn'),
                                         ConvBlock(self.latent_size[4] + self.latent_size[3], self.latent_size[4], kernel_size=3, relu=True, norm='bn'),
                                         ConvBlock(self.latent_size[4], self.latent_size[4], kernel_size=3, relu=True, norm='bn'),
                                         ])
        self.uv_head = ConvBlock(self.latent_size[4], 21, kernel_size=3, padding=1, relu=False, norm=None)
                
        # this is memory comsuming, consider to delete!!!
        # _, self.latent_size = self.get_backbone("resnet50")
        # self.uv_delayer2 = nn.ModuleList([ConvBlock(self.latent_size[2] + self.latent_size[1], self.latent_size[2], kernel_size=3, relu=True, norm='bn'),
        #                                  ConvBlock(self.latent_size[3] + self.latent_size[2], self.latent_size[3], kernel_size=3, relu=True, norm='bn'),
        #                                  ConvBlock(self.latent_size[4] + self.latent_size[3], self.latent_size[4], kernel_size=3, relu=True, norm='bn'),
        #                                  ConvBlock(self.latent_size[4], self.latent_size[4], kernel_size=3, relu=True, norm='bn'),
        #                                  ])
        # self.uv_head2 = ConvBlock(self.latent_size[4], self.uv_channel+1, kernel_size=3, padding=1, relu=False, norm=None)

        reduce_in_channel = 64 + self.uv_channel + len(self.relation)
        self.reduce = nn.Sequential(ConvBlock(reduce_in_channel, reduce_in_channel, relu=True, norm='bn'),
                                    ConvBlock(reduce_in_channel, 128, relu=True, norm='bn'),
                                    ConvBlock(128, 64, kernel_size=1, padding=0, relu=False, norm='bn'))

        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        if self.opt.arch == 'csp_18':
            self.p3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
            self.p4 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1)
            self.p5 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=4, padding=0)
        else: # csp_50
            self.p3 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
            self.p4 = nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1)
            self.p5 = nn.ConvTranspose2d(2048, 256, kernel_size=4, stride=4, padding=0)

        nn.init.xavier_normal_(self.p3.weight)
        nn.init.xavier_normal_(self.p4.weight)
        nn.init.xavier_normal_(self.p5.weight)
        nn.init.constant_(self.p3.bias, 0)
        nn.init.constant_(self.p4.bias, 0)
        nn.init.constant_(self.p5.bias, 0)

        self.p3_l2 = L2Norm(256, 10)
        self.p4_l2 = L2Norm(256, 10)
        self.p5_l2 = L2Norm(256, 10)

        self.feat = nn.Conv2d(768, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.feat_bn = nn.BatchNorm2d(256, momentum=0.01)
        self.feat_act = nn.ReLU(inplace=True)

        # self.depthwise = nn.Conv2d(257, 257, kernel_size=3, padding=1, groups=257, bias=False)
        # self.depthwise.weight.data = self.depthwise.weight.data * 0 + 1/9

        for head in sorted(self.heads):
            num_output = self.heads[head]
            # textures = _tranpose_and_gather_feat(output['texture'], batch['ind'])
            if head_conv > 0:
                if 'params' in head and opt.iterations:
                    extra_chanel = num_output
                else:
                    extra_chanel = 0
                fc = nn.Sequential(
                    nn.Conv2d(256 + extra_chanel, head_conv,
                            kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, num_output,
                            kernel_size=1, stride=1, padding=0))
            else:
                fc = nn.Sequential(
                    nn.Conv2d(
                    in_channels=256,
                    out_channels=num_output,
                    kernel_size=1,
                    stride=1,
                    padding=0
                ))

            if 'hm' in head or 'heatmaps' in head or 'handmap' in head:
                fc[-1].bias.data.fill_(-4.59)
            else:
                fill_fc_weights(fc)
            self.__setattr__(head, fc)
    
    def get_backbone(self, backbone, pretrained=True):
        if '50' in backbone:
            basenet = resnet50(pretrained=pretrained)
            latent_channel = (1000, 2048, 1024, 512, 256)
        elif '18' in backbone:
            basenet = resnet18(pretrained=pretrained)
            latent_channel = (1000, 512, 256, 128, 64)
        else:
            raise Exception("Not supported", backbone)

        return basenet, latent_channel

    def uv_decoder(self, z):
        x = z[0]
        for i, de in enumerate(self.uv_delayer):
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            if i < 3:
                x = torch.cat((x, z[i+1]), dim=1)
            x = de(x)
        pred = _sigmoid(self.uv_head(x))

        return pred

    def uv_decoder2(self, z):
        x = z[0]
        for i, de in enumerate(self.uv_delayer2):
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            if i < 3:
                x = torch.cat((x, z[i+1]), dim=1)
            x = de(x)
        pred = torch.sigmoid(self.uv_head2(x))

        return pred

    def make_conv_layers(self, feat_dims, kernel=3, stride=1, padding=1, bnrelu_final=True):
        layers = []
        for i in range(len(feat_dims)-1):
            layers.append(
                nn.Conv2d(
                    in_channels=feat_dims[i],
                    out_channels=feat_dims[i+1],
                    kernel_size=kernel,
                    stride=stride,
                    padding=padding
                    ))
            # Do not use BN and ReLU for final estimation
            if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and bnrelu_final):
                layers.append(nn.BatchNorm2d(feat_dims[i+1]))
                layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def normal_init(module, mean=0, std=1, bias=0):
        nn.init.normal_(module.weight, mean, std)
        if hasattr(module, 'bias'):
            nn.init.constant_(module.bias, bias)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dilation=dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def make_conv_layers(self, feat_dims, kernel=3, stride=1, padding=1, bnrelu_final=True):
        layers = []
        for i in range(len(feat_dims)-1):
            layers.append(
            nn.Conv2d(
            in_channels=feat_dims[i],
            out_channels=feat_dims[i+1],
            kernel_size=kernel,
            stride=stride,
            padding=padding
            ))
        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and bnrelu_final):
            layers.append(nn.BatchNorm2d(feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def _upsample_concat(self, x, y):
        _,_,H,W = y.size()
        return torch.cat([y, F.upsample(x, size=(H, W), mode='bilinear')], 1)

    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def forward(self, x, heatmaps = None, ind = None): #[b,3,384,384]
        if self.opt.heatmaps:
            # estimate heatmaps first both in training and test
            z_uv = self.backbone(x)
            uv_prior = self.uv_decoder(z_uv[1:])
            x0 = torch.cat([z_uv[0], uv_prior]+ [uv_prior[:, i].sum(dim=1, keepdim=True) for i in self.relation], 1)
            # x0 = torch.cat([z_uv[0], heatmaps],1) # or use the GT heatmaps in training
            x = self.reduce(x0)
            x = self.maxpool(x)
        else:
            x = self.conv1(x) # x[b, 64, 192, 192]
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x) # x[b, 64, 96, 96]

        x1 = self.layer1(x) # x[b, 256, 96, 96]

        x2 = self.layer2(x1) # x[b, 512, 48, 48]
        p3 = self.p3(x2) #[b, 256, 48, 48]
        p3 = self.p3_l2(p3)

        x3 = self.layer3(x2) # x[b, 1024, 24, 24]
        p4 = self.p4(x3) #[b, 256, 48, 48]
        p4 = self.p4_l2(p4)

        x4 = self.layer4(x3) # x[b, 2048, 12, 12]
        p5 = self.p5(x4) #[b, 256, 48, 48]
        p5 = self.p5_l2(p5)

        cat = torch.cat([p3, p4, p5], dim=1) #[b, 768, 48, 48]

        feat = self.feat(cat) #[b, 256, 48, 48]
        feat = self.feat_bn(feat)
        feat = self.feat_act(feat)

        ret = {}
        # decode uv map again for fine result
        if self.opt.heatmaps:
            # z_mesh = [x4,x3,x2,x1]
            # uv_pred = self.uv_decoder2(z_mesh[:])        
            ret['uv_prior'] = uv_prior
            # ret['uv_pred'] = uv_pred[:, :self.uv_channel]
            # ret['mask_pred'] = uv_pred[:, self.uv_channel]
        for head in self.heads:
            if 'hm' in ret and ind is None:
                hms = ret['hm'].clone().detach()
                hms.sigmoid_()
                score = 0.5
                hms = _nms(hms, 5)
                K = max(int((hms > score).float().sum()),1)
                if self.opt.input_res == 512 or self.opt.input_res == 384:
                    K = min(K,10)
                else:
                    K = 1     
                topk_scores, ind, topk_ys, topk_xs = _topk(hms, K)
            if 'params' in head:
                # do iterations for pose params
                thetas = []
                theta = self.mean_theta
                for _ in range(self.iterations):
                    if self.opt.iterations:
                        total_inputs = torch.cat([feat, theta], 1)
                    else:
                        total_inputs = feat
                    theta = theta + self.__getattr__(head)(total_inputs)
                    thetas.append(theta)
                ret[head] = thetas
                continue              
            ret[head] = self.__getattr__(head)(feat)

        return [ret], ind

    def init_weights(self, num_layers, caffe_model=False):
        url = model_urls['resnet{}'.format(num_layers)]
        pretrained_state_dict = model_zoo.load_url(url)
        print('=> loading pretrained model {}'.format(url))
        self.load_state_dict(pretrained_state_dict, strict=False)

            
    def make_deconv_layers(self, feat_dims, bnrelu_final=True):
        layers = []
        for i in range(len(feat_dims)-1):
            if i == len(feat_dims)-2:
                ss = 2
                pp = 1
            else:
                ss = 4
                pp = 0
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=feat_dims[i],
                    out_channels=feat_dims[i+1],
                    kernel_size=4,
                    stride=ss,
                    padding=pp,
                    output_padding=0,
                    bias=False))

            # Do not use BN and ReLU for final estimation
            if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and bnrelu_final):
                layers.append(nn.BatchNorm2d(feat_dims[i+1]))
                layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

# res50 needs to be modified for caltech
resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}


def get_pose_net(num_layers, heads, head_conv=256, down_ratio=8, opt=None):
  block_class, layers = resnet_spec[num_layers]

  model = PoseResNet(block_class, layers, heads, head_conv=head_conv, opt=opt)
  model.init_weights(num_layers)
  return model

def _topk(scores, K):
    b, c, h, w = scores.size()
    assert c == 1
    topk_scores, topk_inds = torch.topk(scores.view(b, -1), K)

    topk_inds = topk_inds % (h * w)
    topk_ys = (topk_inds // w).int().float()
    topk_xs = (topk_inds % w).int().float()
    return topk_scores, topk_inds, topk_ys, topk_xs

def _nms(heat, kernel=5):
    pad = (kernel - 1) // 2
    if kernel == 2:
        hm_pad = F.pad(heat, [0, 1, 0, 1])
        hmax = F.max_pool2d(hm_pad, (kernel, kernel), stride=1, padding=pad)
    else:
        hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep    

