#!/usr/bin/env sh

#train with learning rate 0.01
./build/tools/caffe train \
    --solver=models/trafficSign_jin/solver_lr1.prototxt

#train with learning rate 0.02
./build/tools/caffe train \
    --solver=models/trafficSign_jin/solver_lr2.prototxt \
    --snapshot=models/trafficSign_jin/caffenet_train_iter_600.solverstate
