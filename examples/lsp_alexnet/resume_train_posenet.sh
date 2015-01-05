#!/usr/bin/env sh

prefix=train-07-december
postfix=finetune-from-nov-12-lsp-resume

mkdir cache
mkdir cache/$prefix

TOOLS=../../build/tools

$TOOLS/caffe train -solver caffenet-pose-solver.prototxt -snapshot pose_caffenet_train_iter_54000.solverstate -gpu 0 2>&1 | tee cache/$prefix/$prefix-$postfix.log