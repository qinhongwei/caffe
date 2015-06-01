#!/usr/bin/env sh

./build/tools/caffe train \
    --solver=models/depth_enhancement2/solver.prototxt \
    --snapshot=models/depth_enhancement2/snapshots-2x/dsrcnn_iter_1970000.solverstate \
    --gpu=2
    #--weights=models/depth_enhancement/snapshots-2x/dsrcnn_iter_2470000.solverstate \
    #--snapshot=models/depth_enhancement2/snapshots-2x-finetune/dsrcnn_iter_2460000.solverstate \
    
    #--weights=models/depth_enhancement/snapshots-2x-finetune/dsrcnn_iter_2460000.solverstate \
    #--weights=models/srcnn/snapshots-2x/srcnn_iter_200000.caffemodel \
