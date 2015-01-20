#!/usr/bin/env sh

./build/tools/caffe train \
    --solver=models/cifar10/solver.prototxt \
    --snapshot=models/cifar10/cifar10_nin_iter_20000.solverstate
