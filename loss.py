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

image_shape = (None,None, 3)

# def gaussian(window_size, sigma):
#     gauss = [exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)]
#     print(gauss)
#     quit()
#     gauss = tf.Tensor(gauss, window_size, dtype=tf.float16)
#     return gauss.gauss.reduce_sum()

# def create_window(window_size, channel):
#     _1D_window = gaussian(window_size, 3.5).expand_dims(1)

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
loss = [perceptual_loss, losses.mean_squared_error]
loss_weights = [70,30]

if __name__ == '__main__':
    # print(gaussian(5, 3.5))