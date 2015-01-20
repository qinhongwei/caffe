#!/usr/bin/env sh

#train with learning rate 0.01
./build/tools/caffe train \
    --solver=models/deepfish_jin/solver_lr1.prototxt \
    --gpu=2

#train with learning rate 0.001
./build/tools/caffe train \
    --solver=models/deepfish_jin/solver_lr2.prototxt \
    --snapshot=models/deepfish_jin/caffenet_train_full_iter_600.solverstate \
    --gpu=2
#train with learning rate 0.0001
#./build/tools/caffe train \
#    --solver=models/deepfish_jin/solver_lr3.prototxt \
#    --snapshot=models/deepfish_jin/caffenet_train_full_iter_1200.solverstate \
#    --gpu=2

