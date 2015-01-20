#!/usr/bin/env sh

./build/tools/caffe train \
    --solver=models/trafficSign_jin/solver.prototxt \
    --snapshot=models/trafficSign_jin/caffenet_train_iter_21000.solverstate
