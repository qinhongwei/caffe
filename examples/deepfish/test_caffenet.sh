#!/usr/bin/env sh

./build/tools/caffe train \
    --solver=models/deepfish_jin/solver_test.prototxt \
    --snapshot=models/deepfish_jin/caffenet_train_full_iter_42000.solverstate
