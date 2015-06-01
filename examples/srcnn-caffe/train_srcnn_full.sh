#!/usr/bin/env sh

#train with learning rate 0.0001
./build/tools/caffe train \
    --solver=models/srcnn/solver.prototxt
    --gpu=2

#train with learning rate 0.00001
./build/tools/caffe train \
    --solver=models/srcnn/solver_lr1.prototxt \
    --snapshot=models/srcnn/snapshots-2x/srcnn_iter_100001.solverstate \
    --gpu=2
#train with learning rate 0.000001
./build/tools/caffe train \
    --solver=models/srcnn/solver_lr2.prototxt \
    --snapshot=models/srcnn/snapshots-2x/srcnn_iter_9000001.solverstate \
    --gpu=2
#train with learning rate 0.00001
#./build/tools/caffe train \
#    --solver=models/srcnn/solver_lr3.prototxt \
#    --snapshot=models/srcnn/snapshots/srcnn_iter_10000000.solverstate \
#    --gpu=2
