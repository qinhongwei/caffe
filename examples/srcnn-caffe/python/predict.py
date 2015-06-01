import numpy as np
import matplotlib.pyplot as plt

# Make sure that caffe is on the python path:
caffe_root = '/home/cv/image-net/caffe/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')
import os

import cv2
import caffe

input_dir = '/media/cv/Data/srcnn/Cropped-g/data/'
gt_dir    = '/media/cv/Data/srcnn/Cropped-g/label/'
project_dir = caffe_root + 'models/srcnn/'

caffe.set_mode_cpu()
net = caffe.Net(project_dir + 'deploy.prototxt',
                project_dir + 'snapshots/srcnn_iter_10000000.caffemodel',
                caffe.TEST)
included_extension = ['bmp']
file_names = [fn for fn in os.listdir(input_dir) if any([fn.endswith(ext) for ext in included_extension])]
for fn in file_names:
    # Inputs
    im = caffe.io.load_image(input_dir + fn, color=False)
    # plt.show()
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    #transformer.set_mean('data',np.load(input_dir + 'image_mean.npy').mean(1))
    transformer.set_transpose('data',(2,0,1))
    transformer.set_raw_scale('data',255)
    
    net.blobs['data'].reshape(1,1,32,32)
    net.blobs['data'].data[...] = transformer.preprocess('data', im)

    out = net.forward()

    # Predict results
    mat = out['recon'][0]
    
    # Ground truth
    gt = caffe.io.load_image(gt_dir + fn, 1)
    
    # Show
    """
    mat = cv2.equalizeHist(mat[0,:,:].astype('uint8'))
    im = cv2.equalizeHist(im[6:-6,6:-6,0].astype('uint8'))
    gt = cv2.equalizeHist((gt[:,:,0]*255).astype('uint8'))
    """
    mat = (mat[0,:,:]).astype('uint8')
    im = im[6:-6,6:-6,0].astype('uint8')
    gt = (gt[:,:,0]*255).astype('uint8')
    diff = gt - mat
    # Print Euclidean_loss_layer output
    print (np.dot(diff.flatten(), diff.flatten())).astype(float) / (mat.shape[0]*mat.shape[1]) / 2

    f, axarr = plt.subplots(2, 4, figsize=(15,8))
    axarr[0, 0].imshow(im, cmap='gray')
    axarr[0, 0].set_title('input')
    axarr[0, 1].imshow(mat, cmap='gray')
    axarr[0, 1].set_title('predict')
    axarr[0, 2].imshow(gt, cmap='gray')
    axarr[0, 2].set_title('groundtruth')
    axarr[1, 0].hist(im.flatten(), 256, range=(0.0,255.0), fc='k', ec='k')
    axarr[1, 1].hist(mat.flatten(), 256, range=(0, 255))
    axarr[1, 2].hist(gt.flatten(), 256, range=(0, 255))
    
    axarr[0, 3].imshow(diff)
    axarr[0, 3].set_title('diff')
    axarr[1, 3].hist((diff).flatten(), 100)
    
    plt.savefig(project_dir + 'results_1000/' + fn[0:-4] + '.png')
    plt.show()
