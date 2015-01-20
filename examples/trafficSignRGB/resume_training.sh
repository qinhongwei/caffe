#!/usr/bin/env sh

./build/tools/caffe train \
    --solver=models/trafficSignRGB_jin/solver.prototxt \
    --snapshot=models/trafficSignRGB_jin/caffenet_train_full_iter_21000.solverstate
