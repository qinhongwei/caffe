#!/usr/bin/env sh

./build/tools/caffe train \
    --solver=models/depth_enhancement/solver.prototxt \
    --snapshot=models/depth_enhancement/snapshots-2x-finetune/dsrcnn_iter_2460000.solverstate \
    --gpu=2
    #
    #--weights=models/srcnn/snapshots-2x/srcnn_iter_200000.caffemodel \
