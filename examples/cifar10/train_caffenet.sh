#!/usr/bin/env sh

./build/tools/caffe train \
    --solver=models/cifar10/solver.prototxt
