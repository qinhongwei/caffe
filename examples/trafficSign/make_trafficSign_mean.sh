#!/usr/bin/env sh
# Compute the mean image from the deepfish training leveldb
# N.B. this is available in data/deepfish

EXAMPLE=/media/cv/Data/trafficSign/caffe

./build/tools/compute_image_mean $EXAMPLE/trafficSign_train_lmdb\
  data/trafficSign/trafficSign_mean.binaryproto

echo "Done."
