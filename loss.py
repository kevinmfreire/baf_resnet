# -*- coding: utf-8 -*-
"""
Created on Thu June 9 20:37 2022
This code uses perceptual loss, dissimilarity index and noise conscious mse for optimization. also shows the metric psnr
loss: vgg16 ['block1_conv2','block2_conv2','block3_conv3','block4_conv3']
@author: kevinmfreire
"""

import numpy as np
import tensorflow as tf

# from keras.models import Model
# from keras.layers import Dense,concatenate, Activation, Lambda
# from keras.layers import Conv2D, add, Input,Conv2DTranspose
# from keras.optimizers import SGD,Adam
from keras import losses
# from keras.preprocessing.image import ImageDataGenerator
# from matplotlib import pyplot as plt
# import math
# import h5py
# from keras.initializers import RandomNormal
# #from preprocess_CT_image import load_scan, get_pixels_hu, write_dicom, map_0_1,windowing2
# from keras.layers import BatchNormalization as BN

from keras import backend as K

from keras.applications.vgg16 import VGG16
from keras.models import Model

from math import exp

class SSIM(object):
    def __init__(self, k1=0.01, k2=0.02, L=1, window_size=11):
        self.k1 = k1
        self.k2 = k2           # constants for stable
        self.L = L             # the value range of input image pixels
        self.WS = window_size

    def _tf_fspecial_gauss(self, size, sigma=1.5):
        """Function to mimic the 'fspecial' gaussian MATLAB function"""
        x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

        x_data = np.expand_dims(x_data, axis=-1)
        x_data = np.expand_dims(x_data, axis=-1)

        y_data = np.expand_dims(y_data, axis=-1)
        y_data = np.expand_dims(y_data, axis=-1)

        x = tf.constant(x_data, dtype=tf.float32)
        y = tf.constant(y_data, dtype=tf.float32)

        g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
        return g / tf.reduce_sum(g)

    def ssim_loss(self, img1, img2):
        """
        The function is to calculate the ssim score
        """
        window = self._tf_fspecial_gauss(size=self.WS)  # output size is (window_size, window_size, 1, 1)
        #import pdb
        #pdb.set_trace()

        (_, _, _, channel) = img1.shape.as_list()

        window = tf.tile(window, [1, 1, channel, 1])

        # here we use tf.nn.depthwise_conv2d to imitate the group operation in torch.nn.conv2d 
        mu1 = tf.nn.depthwise_conv2d(img1, window, strides = [1, 1, 1, 1], padding = 'VALID')
        mu2 = tf.nn.depthwise_conv2d(img2, window, strides = [1, 1, 1, 1], padding = 'VALID')

        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        img1_2 = img1*img1#tf.pad(img1*img1, [[0,0], [0, self.WS//2], [0, self.WS//2], [0,0]], "CONSTANT")
        sigma1_sq = tf.subtract(tf.nn.depthwise_conv2d(img1_2, window, strides = [1 ,1, 1, 1], padding = 'VALID') , mu1_sq)
        img2_2 = img2*img2#tf.pad(img2*img2, [[0,0], [0, self.WS//2], [0, self.WS//2], [0,0]], "CONSTANT")
        sigma2_sq = tf.subtract(tf.nn.depthwise_conv2d(img2_2, window, strides = [1, 1, 1, 1], padding = 'VALID') ,mu2_sq)
        img12_2 = img1*img2#tf.pad(img1*img2, [[0,0], [0, self.WS//2], [0, self.WS//2], [0,0]], "CONSTANT")
        sigma1_2 = tf.subtract(tf.nn.depthwise_conv2d(img12_2, window, strides = [1, 1, 1, 1], padding = 'VALID') , mu1_mu2)

        c1 = (self.k1*self.L)**2
        c2 = (self.k2*self.L)**2

        ssim_map = ((2*mu1_mu2 + c1)*(2*sigma1_2 + c2)) / ((mu1_sq + mu2_sq + c1)*(sigma1_sq + sigma2_sq + c2))

        return tf.reduce_mean(ssim_map)

def perceptual_loss(y_true, y_pred):
    vgg = VGG16(include_top=False, weights='imagenet', input_shape=image_shape)
    selectedLayers = ['block1_conv2','block2_conv2','block3_conv3','block4_conv3']
    selectedOutputs = [vgg.get_layer(i).output for i in selectedLayers]
    loss_model = Model(inputs=vgg.input, outputs=selectedOutputs)
    loss_model.trainable = False
    mse = K.variable(value=0)
    for i in range(0,3):
        mse = mse+ K.mean(K.square(loss_model(y_true)[i] - loss_model(y_pred)[i]))
    return mse
#
#model_edge_p.load_weights('Weights/weights_DRL_sobel4d_adam1_perceptual_mse_th.h5')
loss = [perceptual_loss, losses.mean_squared_error, SSIM.ssim_loss]
loss_weights = [40,20,40]

if __name__ == '__main__':
    # print(gaussian(5, 3.5))
    image_shape = (None,None, 3)