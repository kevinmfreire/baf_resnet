# -*- coding: utf-8 -*-
"""
Created on Fri June 10 15:23 2022
@author:kevinmfreire
Based on: https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial
"""
import os
import argparse
import numpy as np
import pydicom
# import scipy.ndimage

def save_dataset(args):
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        print('Create path : {}'.format(args.save_path))

    patients_list = ['L019', 'L033', 'L058']#,  'L064', 'L077', 'L114', 'L116', 'L134', 'L145', 'L150']
    for p_ind, patient in enumerate(patients_list):
        patient_input_path = os.path.join(args.data_path, patient,
                                          "quarter_{}mm".format(args.mm))
        patient_target_path = os.path.join(args.data_path, patient,
                                           "full_{}mm".format(args.mm))

        for path_ in [patient_input_path, patient_target_path]:
            full_pixels = get_pixels_hu(load_scan(path_))
            for pi in range(len(full_pixels)):
                io = 'input' if 'quarter' in path_ else 'target'
                f = full_pixels[pi]
                f_name = '{}_{}_{}.npy'.format(patient, pi, io)
                np.save(os.path.join(args.save_path, f_name), f)

        printProgressBar(p_ind, len(patients_list),
                         prefix="save image ..",
                         suffix='Complete', length=25)
        print(' ')


def load_scan(path):
    # referred from https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial
    slices = [pydicom.read_file(os.path.join(path, s)) for s in os.listdir(path)]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
    for s in slices:
        s.SliceThickness = slice_thickness
    return slices


def get_pixels_hu(slices):
    # referred from https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)
    image[image == -2000] = 0
    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
        image[slice_number] += np.int16(intercept)
    return normalize_(np.array(image, dtype=np.int16))


def normalize_(image, MIN_B=-1024.0, MAX_B=3072.0):
   image = (image - MIN_B) / (MAX_B - MIN_B)
   return image

def denormalize_(image, MIN_B=-1024.0, MAX_B=3072.0):
    image = (image * (MAX_B - MIN_B)) + MIN_B
    return image

def normalization(image):
    mean_image = np.mean(image, axis = 0).astype(np.float32)
    std_image = np.std(image,axis = 0).astype(np.float32)
    out = ((image-mean_image)/std_image).astype(np.float32)
    out = np.nan_to_num(out)
    return out

# To have similar thickness when using different datasets
# def resample(image, scan, new_spacing=[1,1,1]):
#     # Determine current pixel spacing
#     spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)

#     resize_factor = spacing / new_spacing
#     new_real_shape = image.shape * resize_factor
#     new_shape = np.round(new_real_shape)
#     real_resize_factor = new_shape / image.shape
#     new_spacing = spacing / real_resize_factor
    
#     image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    
#     return image, new_spacing

#  map array between zero and 1 , find max and min
def map_0_1(array):
    out = np.zeros(array.shape)
    #    max_out = np.zeros(array.shape[0])
    #    min_out = np.zeros(array.shape[0])

    for n,val in enumerate(array):
        out[n] = (val-val.min())/(val.max()-val.min())
    #        max_out[n] = val.max()
    #        min_out[n] = val.min()
    
    out = np.nan_to_num(out)
    return out.astype(np.float32)#,max_out,min_out

def remove_padding(slices):
    # read the dicom images, remove padding, create 4D matrix

    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    try:
        padding = slices[0].PixelPaddingValue
    except:
        padding = 0
    
    image[image == padding] = 0
    
    return np.array(image, dtype=np.int16)

def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill=' '):
    # referred from https://gist.github.com/snakers4/91fa21b9dda9d055a02ecd23f24fbc3d
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '=' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    if iteration == total:
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='../ldct-denoising/patient/')
    parser.add_argument('--save_path', type=str, default='./processed_data/npy_img/')

    parser.add_argument('--test_patient', type=str, default='L058')
    parser.add_argument('--mm', type=int, default=3)
    parser.add_argument('--norm_range_min', type=float, default=-1024.0)
    parser.add_argument('--norm_range_max', type=float, default=3072.0)

    args = parser.parse_args()
    save_dataset(args)