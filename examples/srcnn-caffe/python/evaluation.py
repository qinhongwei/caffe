import numpy as np
import matplotlib.pyplot as plt
import cv2

# Make sure that caffe is on the python path:
caffe_root = '/home/cv/image-net/caffe/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')
import os

import caffe

zooming = 3

#for display
'''def colorize(y, gt): #y is the predicted Y image ,gt is the groundtruth color image
    y[y>255] = 255
    
    ycrcb = cv2.cvtColor(gt,cv2.COLOR_RGB2YCR_CB)
    img = np.zeros((y.shape[0], y.shape[1], 3), np.uint8)
    //upscaling
    ycrcb[:,:,1] = cv2.resize(ycrcb[:,:,1], (int(y.shape[0]/zooming), int(y.shape[1]/zooming)))
    ycrcb[:,:,1] = cv2.resize(ycrcb[:,:,1], (y.shape[0], y.shape[1]))
    ycrcb[:,:,2] = cv2.resize(ycrcb[:,:,2], (int(y.shape[0]/zooming), int(y.shape[1]/zooming)))
    ycrcb[:,:,2] = cv2.resize(ycrcb[:,:,2], (y.shape[0], y.shape[1]))
    img[:,:,0] = y
    img[:,:,1] = ycrcb[:,:,1]
    img[:,:,2] = ycrcb[:,:,2]
    img = cv2.cvtColor(img, cv2.COLOR_YCR_CB2BGR)
    
    return img

# PSNR measure, from ANR's code
def PSNR(pred, gt): #pred is the predicted Y image ,gt is the groundtruth color image
    f = pred.astype(float)
    #just for the Y channel
    gt_ycrcb = cv2.cvtColor(gt,cv2.COLOR_RGB2YCR_CB)
    gt_y = gt_ycrcb[:,:,0]
    g = gt_y.astype(float)
    e = (f - g).flatten()
    n = len(e)
    rst = 10*np.log10(n/e.dot(e))
    
    return rst'''
def colorize(y, ycrcb):
    y[y>255] = 255
    
    img = np.zeros((y.shape[0], y.shape[1], 3), np.uint8)
    img[:,:,0] = y
    img[:,:,1] = ycrcb[:,:,1]
    img[:,:,2] = ycrcb[:,:,2]
    img = cv2.cvtColor(img, cv2.COLOR_YCR_CB2BGR)
    
    return img

# PSNR measure, from ANR's code
def PSNR(pred, gt):
    f = pred.astype(float)
    g = gt.astype(float)
    e = (f - g).flatten()
    n = len(e)
    rst = 10*np.log10(n/e.dot(e))
    
    return rst
