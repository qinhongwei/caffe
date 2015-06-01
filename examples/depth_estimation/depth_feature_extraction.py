import numpy as np
import matplotlib.pyplot as plt

# Make sure that caffe is on the python path:
caffe_root = '/home/cv/image-net/caffe/'  # the caffe-root
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

import os
if not os.path.isfile(caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'):
    print "Downloading pre-trained CaffeNet model..." 

caffe.set_mode_gpu()
net = caffe.Net(caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt',
                caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
                caffe.TEST)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

net.blobs['data'].reshape(1,3,227,227) #change to the adapt to the net
#read the image index file
index = open('examples/depth_estimation/Train400ImgIndex.txt')
img_index = index.readlines()  #the image name
index.close()
featfile = 'examples/depth_estimation/train400_features.txt'
if os.path.isfile(featfile):
    os.remove(featfile)

myfile = 'examples/depth_estimation/train400_predcls_alexnet.txt'
if os.path.isfile(myfile):
    os.remove(myfile)

for idx in img_index:
    num = 0
    idx = idx[:-1]  #remove the '\n'
    img_name = idx.split('/')[-1]
    try:
        net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(idx)) #idx:the absolute path of images
    except ValueError:
        print "can not convert object to float64."
        continue
          
    out = net.forward()  #the forward process to get the result
    pred_class = open(myfile,'a')
    pred_class.write(str(out['prob'].argmax())+'\n')
    
    print("Predicted class is #{}.".format(out['prob'].argmax()))
 
    feat = net.blobs['fc7'].data[0] #get the features we want 1*4096 row vector
    output = open(featfile,'a')
    '''if num ==1:
        output.write(str(img_name)+'\n')
    else:
        output.write('\n'+str(img_name)+'\n')'''
    for i in feat:
        num = num+1
        if num == feat.size:
            output.write(str(i)+'\n')
        else:
            output.write(str(i)+'\t')
    output.close()
    pred_class.close()




















