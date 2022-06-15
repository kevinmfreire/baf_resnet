# -*- coding: utf-8 -*-
"""
@author: kevinmfreire
"""
#Model Components
# import tensorflow as tf
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Conv2D, Multiply, Conv2DTranspose
# from tensorflow.keras.layers import Activation, AveragePooling2D
# from tensorflow.keras.layers import Add, concatenate, Input
# from tensorflow.keras.layers import BatchNormalization as BN


import torch.nn as nn
import torch
import torch.nn.functional as F
from math import exp
from torch.autograd import Variable
import numpy as np
from typing import Callable, Any, Optional, Tuple, List
import warnings
from torchsummary import summary

class spatial_attention(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding='valid'):
        super(spatial_attention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.final_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv4 = torch.mul(conv1, conv2)
        conv4 = self.softmax(conv4)
        conv5 = torch.mul(conv3, conv4)
        conv5 = self.final_conv(conv5)
        sa_feat_map = torch.add(x, conv5)
        return sa_feat_map

class channel_attention(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding='valid'):
        super(channel_attention, self).__init__()
        self.cam = nn.Sequential(nn.AvgPool2d(1, stride=1),
                                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                                nn.ReLU(),
                                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                                nn.Sigmoid())
    
    def forward(self, x):
        out = self.cam(x)
        return torch.mul(x, out)

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same', dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=True)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x)

class bafnet(nn.Module):
    def __init__(self, conv_block: Optional[Callable[..., nn.Module]] = None):
        super(bafnet, self).__init__()

        self.in_channels = 64

        if conv_block is None:
            block = [BasicConv2d, spatial_attention, channel_attention]

        assert len(block) == 3
        conv_block = block[0]
        sam = block[1]
        cam = block[2]

        # 1) Preconvolutional module (number of filters in each layer = 1)
        self.conv_1 = conv_block(1, 64, 3, dilation=2)
        self.conv_2 = conv_block(64, 64, 3, dilation=2)

        # 2) Initialize the channel and spatial Attention Modules
        self.SAM = sam(64, 64, 1, 1)
        self.CAM = cam(64, 64, 1, 1)

        # 3) Initialize the convolutional layers for the Boosting Attention Fusion Block (BAFB)
        self.conv_relu = nn.Sequential(nn.Conv2d(self.in_channels, self.in_channels, 1, 1, padding='valid'),
                                    nn.ReLU())
        self.cat_conv1 = nn.Sequential(nn.Conv2d(self.in_channels*4, 1, 1, 1, padding='valid'),
                                        nn.ReLU())
        self.cat_conv2 = nn.Conv2d(self.in_channels+1, self.in_channels, 1, 1, padding='valid')
        # self.conv = nn.Conv2d(1, self.in_channels, 1, 1, padding='valid')

        # 4) Post convolution module
        self.deconv = conv_block(64, 64, 3, dilation=2)

        # 5) Reconstruction Layer
        self.out = nn.ConvTranspose2d(64, 1, 3, stride=1, dilation=2, padding=2)

    def forward(self, x):
        # Section 1
        conv1 = self.conv_1(x)
        conv2 = self.conv_2(conv1)
        conv3 = self.conv_2(conv2)

        # Sections 2 and 3
        bmg1 = self.BMG(conv3)
        bmg2 = self.BMG(bmg1)

        # Section 4
        skip1 = torch.add(conv3, bmg2)
        deconv1 = self.deconv(skip1)
        skip2 = torch.add(conv2, deconv1)
        deconv2 = self.deconv(skip2)
        skip3 = torch.add(conv1, deconv2)
        deconv3 = self.deconv(skip3)

        # Section 5
        out = self.out(deconv3)
        
        return out

    def BAFB(self, input):

        conv1 = self.conv_relu(input)

        sam1 = self.SAM(conv1)
        sam1 = torch.add(sam1, conv1)
        cam1 = self.CAM(sam1)

        cam2 = self.CAM(conv1)
        cam2 = torch.add(cam2, conv1)
        sam2 = self.SAM(cam2)

        fuse1 = self.cat_conv1(torch.cat([cam1, sam1, sam2, cam2], 1))
        fuse2 = self.cat_conv2(torch.cat([conv1, fuse1],1))

        # out = self.conv(fuse2)

        return fuse2

    def BMG(self, input):
        bafb1 = self.BAFB(input)
        bafb2 = self.BAFB(bafb1)
        bafb3 = self.BAFB(bafb2)
        bafbn = self.BAFB(bafb3)
        out = torch.add(input, bafbn)
        return out

# -------------------------------End---------------------------------------------

if __name__ == '__main__':

    input=(1, 120, 120)
    baf_net = bafnet()
    summary(baf_net, input)