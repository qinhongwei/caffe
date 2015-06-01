import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Make sure that caffe is on the python path:
caffe_root = '/home/cv/image-net/caffe/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')
sys.path.insert(0, caffe_root + 'examples/srcnn-caffe/python')

import caffe
from evaluation import *

# Parameters
zooming = 2

input_dir = '/media/cv/Data/srcnn/Cropped-g/data/'
gt_dir    = '/media/cv/Data/srcnn/Cropped-g/label/'
project_dir = caffe_root + 'models/srcnn/'

caffe.set_mode_cpu()

'''net = caffe.Net(caffe_root + 'models/srcnn/SRCNN_deploy_1bmp.prototxt',
                caffe_root + 'models/srcnn/snapshots-2x/srcnn_iter_10000000.caffemodel',
                caffe.TEST)
im_raw = cv2.imread('/media/cv/Data/srcnn/Training/t1.bmp')'''

net = caffe.Net(caffe_root + 'models/srcnn/SRCNN_deploy_Set5.prototxt',
                caffe_root + 'models/srcnn/snapshots-2x/srcnn_iter_200000.caffemodel', #the test ouput is lower, the PSNR is higher
                caffe.TEST)
im_raw = cv2.imread('/media/cv/Data/srcnn/Test/Set5/baby_GT.bmp')

'''net = caffe.Net(caffe_root + 'models/srcnn/SRCNN_deploy_Set14.prototxt',
                caffe_root + 'models/srcnn/snapshots-2x/srcnn_iter_1000001.caffemodel',
                caffe.TEST)
im_raw = cv2.imread('/media/cv/Data/srcnn/Test/Set14/pepper.bmp')'''

ycrcb = cv2.cvtColor(im_raw, cv2.COLOR_RGB2YCR_CB)
im_raw = ycrcb[:,:,0] #y
# im_raw = im_raw.reshape((im_raw.shape[0], im_raw.shape[1], 1))

im_small = cv2.resize(im_raw, (int(im_raw.shape[1]/zooming), int(im_raw.shape[0]/zooming)))
im_blur = cv2.resize(im_small, (im_raw.shape[1], im_raw.shape[0]))
im_blur = im_blur.reshape(im_blur.shape[0], im_blur.shape[1], 1)

im_input = im_blur
#plt.show()
#image_mean = np.load(input_dir + 'image_mean.npy')
# net.set_mean('data', image_mean)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data',(2,0,1))
transformer.set_raw_scale('data',255)

net.blobs['data'].data[...] = transformer.preprocess('data', im_input.astype(float)/255)
out = net.forward()
# Predict results
mat = out['recon'][0]
#print mat.shape
# Show
ycrcb = ycrcb[6:-6,6:-6,:]
im_pred = colorize(mat[0,:,:], ycrcb)
im_input = colorize(im_input[6:-6,6:-6,0], ycrcb)
im_raw = colorize(im_raw[6:-6,6:-6], ycrcb)

# PSNR
psnr = PSNR(im_pred.astype(float)/255, im_raw.astype(float)/255)
print psnr
ss
f, arr = plt.subplots(2, 2)
arr[0][0].imshow(im_raw)
arr[0][0].set_title("raw")
arr[0][1].imshow(im_input)
arr[0][1].set_title("input")
arr[1][0].imshow(im_pred)
arr[1][0].set_title('predict')
diff = im_raw - im_pred
arr[1][1].imshow(diff)
arr[1][1].set_title('diff')
plt.xlabel('PSNR: '+str(psnr))
plt.savefig(project_dir + 'results_100/butterfly_blur_%dx.png' % zooming)
plt.show()
