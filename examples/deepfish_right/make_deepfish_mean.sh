#!/usr/bin/env sh
# Compute the mean image from the deepfish training leveldb
# N.B. this is available in data/deepfish

EXAMPLE=/media/cv/Data/fishData/fish_boundingbox_47_right_caffe

~/image-net/caffe/build/tools/compute_image_mean $EXAMPLE/deepfish_train_lmdb\
  ../../data/deepfish/deepfish_mean_right.binaryproto

echo "Done."
