#!/usr/bin/env sh

./build/tools/caffe train \
    --solver=models/deepfish_jin_right/solver_lr2.prototxt \
    --snapshot=models/deepfish_jin_right/caffenet_train_full_iter_90000.solverstate \
    --gpu=0
    #--weights=models/deepfish_jin_right/caffenet_train_full_iter_45000.caffemodel
