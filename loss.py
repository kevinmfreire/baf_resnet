# -*- coding: utf-8 -*-
"""
Created on Thu June 9 20:37 2022
This code uses perceptual loss, dissimilarity index and noise conscious mse for optimization. also shows the metric psnr
loss: vgg16 ['block1_conv2','block2_conv2','block3_conv3','block4_conv3']
@author: kevinmfreire
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
import argparse
from math import exp
from torch.autograd import Variable
from torch import linalg
import torchvision.models as models
from torchvision import transforms
from torchvision.models._utils import IntermediateLayerGetter
from torch.nn.modules.loss import _Loss

# Check CUDA's presence
cuda_is_present = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda_is_present else torch.FloatTensor

def to_cuda(data):
    	return data.cuda() if cuda_is_present else data

def normalize_(image, MIN_B=-1024.0, MAX_B=3072.0):
    image = (image - MIN_B) / (MAX_B - MIN_B)
    return image

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window =_1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return  to_cuda(window)

# OBTAIN VGG16 PRETRAINED MODEL EXCLUDING FULLY CONNECTED LAYERS
def get_feature_layer_vgg16(image, layer, model):
    image = torch.cat([image,image,image],1)
    return_layers = {'{}'.format(layer): 'feat_layer_{}'.format(layer)}
    output_feature = IntermediateLayerGetter(model.features, return_layers=return_layers)
    image_feature = output_feature(image)
    return image_feature['feat_layer_{}'.format(layer)]

def compute_SSIM(img1, img2, window_size, channel, size_average=True):
    # referred from https://github.com/Po-Hsun-Su/pytorch-ssim
    if len(img1.size()) == 2:
        shape_ = img1.shape[-1]
        img1 = img1.view(1,1,shape_ ,shape_ )
        img2 = img2.view(1,1,shape_ ,shape_ )
    window = create_window(window_size, channel)
    window = window.type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size//2)
    mu2 = F.conv2d(img2, window, padding=window_size//2)
    mu1_sq, mu2_sq = mu1.pow(2), mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2) - mu1_mu2

    C1, C2 = 0.01**2, 0.03**2

    ssim_map = ((2*mu1_mu2+C1)*(2*sigma12+C2)) / ((mu1_sq+mu2_sq+C1)*(sigma1_sq+sigma2_sq+C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class Vgg16FeatureExtractor(nn.Module):
    
    def __init__(self, layers=[3, 8, 15, 22, 29],pretrained=False, progress=True, **kwargs):
        super(Vgg16FeatureExtractor, self).__init__()
        self.layers=layers
        self.model = models.vgg16(pretrained, progress, **kwargs)
        del self.model.avgpool
        del self.model.classifier
        self.return_layers = {'{}'.format(self.layers[i]): 'feat_layer_{}'.format(self.layers[i]) for i in range(len(self.layers))}
        self.model = IntermediateLayerGetter(self.model.features, return_layers=self.return_layers)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

    def forward(self, x):
        feats = list()
        # x = self.normalize(x)
        out = self.model(x)
        for i in range(len(self.layers)):
            feats.append(out['feat_layer_{}'.format(self.layers[i])])
        return feats

class SSIM(nn.Module):
    """
    The Dissimilarity Loss funciton
    """
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)
        self.window.to(torch.device('cuda' if cuda_is_present else 'cpu'))

    def forward(self, pred, target):
        # target, pred = denormalize_(target), denormalize_(pred)
        ssim = compute_SSIM(target, pred, self.window_size, self.channel, self.size_average)
        dssim = (1.0-ssim)
        return dssim

class CompoundLoss(nn.Module):
    
    def __init__(self, blocks=[1, 2, 3, 4, 5], vgg_weight=0.3, ssim_weight=0.5, mse_weight=0.2):
        super(CompoundLoss, self).__init__()

        self.vgg_weight = vgg_weight
        self.ssim_weight = ssim_weight
        self.mse_weight = mse_weight

        self.blocks = blocks
        self.model = Vgg16FeatureExtractor(pretrained=True)

        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()

        self.mse = nn.MSELoss()
        self.ssim = SSIM()

    def forward(self, pred, ground_truth):
        loss_value = 0

        input_feats = self.model(torch.cat([pred, pred, pred], dim=1))
        target_feats = self.model(torch.cat([ground_truth, ground_truth, ground_truth], dim=1))

        feats_num = len(self.blocks)
        for idx in range(feats_num):
            input, target = input_feats[idx], target_feats[idx]
            loss_value += self.mse(input, target)

        loss_value /= feats_num
        ssim_loss = self.ssim(pred, ground_truth)
        mse_loss = self.mse(pred, ground_truth)
        
        loss = self.vgg_weight * loss_value + self.ssim_weight * ssim_loss + self.mse_weight * mse_loss

        return loss