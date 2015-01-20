#!/usr/bin/env sh

#train with learning rate 0.01
./build/tools/caffe train \
    --solver=models/trafficSignRGB_jin/solver_lr1.prototxt

#train with learning rate 0.001
./build/tools/caffe train \
    --solver=models/trafficSignRGB_jin/solver_lr2.prototxt \
    --snapshot=models/trafficSignRGB_jin/caffenet_train_full_iter_600.solverstate
