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

class spatial_attention():
    def __init__(self):
        super(spatial_attention, self).__init__()
        self.conv1 = nn.Conv2d(64, 64, 1, 1, padding='valid')
        self.conv2 = nn.Conv2d(64, 64, 1, 1, padding='valid')
        self.conv3 = nn.Conv2d(64, 64, 1, 1, padding='valid')
        self.final_conv = nn.Conv2d(64, 64, 1, 1, padding='valid')

    def forward(self, x):

        return 


# Spatial Attention
def SA(feat_map):
    conv1 = Conv2D(64, (1,1), strides=1, padding='valid')(feat_map)
    conv2 = Conv2D(64, (1,1), strides=1, padding='valid')(feat_map)
    conv3 = Conv2D(64, (1,1), strides=1, padding='valid')(feat_map)
    
    conv4 = Multiply()([conv1, conv2])
    conv5 = Multiply()([conv3, conv4])
    conv5 = Conv2D(64, (1,1), strides=1, padding='valid')(conv5)
    
    sa_feat_map = Add()([feat_map, conv5])
    
    return sa_feat_map

# Channel Attention 
def CA(feat_map):
    conv1 = AveragePooling2D(pool_size=(1,1), strides=1, padding='valid')(feat_map)  
    conv2 = Conv2D(64, (1,1), strides=1, activation='relu', padding='valid')(conv1)
    #conv2 = Activation('relu')(conv2)
    
    conv3 = Conv2D(64, (1,1), strides=1, activation='sigmoid',padding='valid')(conv2)
    #conv3 = Activation('sigmoid')(conv3)
    
    ca_feat_map = Multiply()([feat_map, conv3])
    
    return ca_feat_map

# BAFB -- four for each BMG in the denoiser
def BAFB(input_bafb):
    fcr1 = Conv2D(64, (1,1), strides=1, activation='relu', padding='valid')(input_bafb)
    
    fsa1 = SA(fcr1)
    fes_up = Add()([fsa1,fcr1])
    
    fca1 = CA(fcr1)
    fes_down = Add()([fca1, fcr1])
    
    fca2 = CA(fes_up)
    fsa2 = SA(fes_down)
    
    fcr2=concatenate([fca2, fes_up, fes_down, fsa2],axis=3)
    fcr2=Conv2D(1, (1,1), strides=1, activation='relu', padding='valid')(fcr2) 
    
    fc=concatenate([fcr1, fcr2],axis=3)
    fc=Conv2D(1, (1,1), strides=1, activation='relu', padding='valid')(fc) 
    
    return fc

# Boosting Module Groups (BMG)
def BMG(bmg_input):
    bafb1 = BAFB(bmg_input)
    bafb2 = BAFB(bafb1)
    bafb3 = BAFB(bafb2)
    bafbn = BAFB(bafb3)   
    fg = Add()([bmg_input,bafbn]) #group skip connection 
    return fg

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same', dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class bafnet(nn.Module):
    def __init__(self, conv_block: Optional[Callable[..., nn.Module]] = None):
        super(bafnet, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d

        self.conv1 = conv_block(1, 64, 3, dilation=2)
        self.conv2 = conv_block(64, 64, 3, dilation=2)
        self.conv3 = conv_block(64, 64, 3, dilation=2)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        add1 = x + conv2
        conv3 = self.conv3(add1)
        return x

def baf_resnet(inputs):
    # inputs=(None, None, 1)
    inputs=Input(shape=inputs)
    # conv_block = BasicConv2d()

    # 1) Preconvolutional module (number of filters in each layer = 1)
    f1sf = Conv2D(64, (3,3), dilation_rate=(2,2), padding='same')(inputs)
    f1sf = BN()(f1sf)
    f1sf = Activation('relu')(f1sf)
    # quit()

    # f1sf = conv_block(inputs)
    
    f2sf = Conv2D(64, (3,3), dilation_rate=(2,2),padding='same')(f1sf)
    f2sf = BN()(f2sf)
    f2sf = Activation('relu')(f2sf)
    
    f3sf = Add()([inputs, f2sf])
    f3sf = Conv2D(64, (3,3), dilation_rate=(2,2),padding='same')(f3sf)
    f3sf = BN()(f3sf)
    f3sf = Activation('relu')(f3sf)
    
    
    # 2) Two BMGs with 4 BAFBs each
    bmg1= BMG(f3sf)
    bmg2= BMG(bmg1)
    
    f4sf=Add()([f3sf, bmg2])
    
    # 3) Post convolution Fpost = R3postC(R2postC( R1postC(bmg2)+F2sf ) + F1sf )
    dconv3 = Conv2D(64, (3,3), dilation_rate=(2,2),padding='same')(f4sf)
    dconv3 = BN()(dconv3)
    dconv3 = Activation('relu')(dconv3)
    
    dconv2 = Add()([f2sf, dconv3]) # Symmetric skip connection (SSC)
    dconv2 = Conv2D(64, (3,3), dilation_rate=(2,2),padding='same')(dconv3)  
    dconv2 = BN()(dconv2)
    dconv2 = Activation('relu')(dconv2)
    
    dconv1 = Add()([f1sf, dconv2])  # Symmetric skip connection (SSC)
    dconv1 = Conv2D(64, (3,3), dilation_rate=(2,2),padding='same')(dconv2)  
    dconv1 = BN()(dconv1)
    dconv1 = Activation('relu')(dconv1)
    
    # 4) Reconstruction Layer
    out= Conv2DTranspose(1, (3,3), dilation_rate=(2,2),padding='same')(dconv1) 
    # out = BN()(out)
    # out = Activation('relu')(out)
    #resnet=Model(inputs=[inputs], outputs=[out, out]) #uncomment for combinations of losses
    resnet=Model(inputs=[inputs], outputs=[out]) 
    return resnet

# -------------------------------End---------------------------------------------

if __name__ == '__main__':

    input=(1, 120, 120)
    # model=baf_resnet(input)
    baf_net = bafnet()
    summary(baf_net, input)
    # model.summary()