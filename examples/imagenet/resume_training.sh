#!/usr/bin/env sh

./build/tools/caffe train \
    --solver=models/bvlc_reference_caffenet/solver.prototxt \
    #--snapshot=models/bvlc_reference_caffenet/caffenet_train_iter_420000.solverstate
    --weights=models/bvlc_reference_caffenet/caffenet_train_iter_420000.caffemodel

    #--weights=models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel
