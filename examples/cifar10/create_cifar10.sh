#!/usr/bin/env sh
# This script converts the cifar data into leveldb format.

EXAMPLES=../../bin
DATA=../../data/cifar10
TOOLS=../../bin

echo "Creating leveldb..."

rm -rf cifar10-leveldb
mkdir cifar10-leveldb

$EXAMPLES/convert_cifar_data.exe $DATA ./cifar10-leveldb

echo "Computing image mean..."

$TOOLS/compute_image_mean.exe ./cifar10-leveldb/cifar-train-leveldb mean.binaryproto

echo "Done."
