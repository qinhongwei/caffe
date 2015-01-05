#!/usr/bin/env sh
# Compute the mean image from the imagenet training leveldb
# N.B. this is available in data/ilsvrc12

EXAMPLE=/media/cv/SSD/imagenet

./build/tools/compute_image_mean $EXAMPLE/ilsvrc12_train_lmdb\
  data/ilsvrc12/imagenet_mean.binaryproto

echo "Done."
