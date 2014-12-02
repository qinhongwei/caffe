#!/usr/bin/env python
# make sure that caffe is on the python path:
caffe_root = '../../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

# Load the original network and extract the fully-connected layers' parameters.
net = caffe.Net(caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt', caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')
params = ['fc6', 'fc7', 'fc8']
# fc_params = {name: (weights, biases)}
fc_params = {pr: (net.params[pr][0].data, net.params[pr][1].data) for pr in params}

for fc in params:
    print '{} weights are {} dimensional and biases are {} dimensional'.format(fc, fc_params[fc][0].shape, fc_params[fc][1].shape)

# Load the fully-convolutional network to transplant the parameters.
net_full_conv = caffe.Net(caffe_root + 'examples/imagenet/bvlc_caffenet_full_conv.prototxt', caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')
params_full_conv = ['fc6-conv', 'fc7-conv', 'fc8-conv']
# conv_params = {name: (weights, biases)}
conv_params = {pr: (net_full_conv.params[pr][0].data, net_full_conv.params[pr][1].data) for pr in params_full_conv}

for conv in params_full_conv:
    print '{} weights are {} dimensional and biases are {} dimensional'.format(conv, conv_params[conv][0].shape, conv_params[conv][1].shape)

for pr, pr_conv in zip(params, params_full_conv):
    conv_params[pr_conv][1][...] = fc_params[pr][1]

for pr, pr_conv in zip(params, params_full_conv):
    out, in_, h, w = conv_params[pr_conv][0].shape
    W = fc_params[pr][0].reshape((out, in_, h, w))
    conv_params[pr_conv][0][...] = W

net_full_conv.save(caffe_root + 'examples/imagenet/bvlc_caffenet_full_conv.caffemodel')
