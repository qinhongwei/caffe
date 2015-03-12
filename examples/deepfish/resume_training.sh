#!/usr/bin/env sh

./build/tools/caffe train \
    --solver=models/deepfish_jin_new/solver_lr2.prototxt \
    --snapshot=models/deepfish_jin_new/caffenet_train_full_iter_39200.solverstate
    #--weights=models/deepfish_jin_new/caffenet_train_full_iter_45000.caffemodel
