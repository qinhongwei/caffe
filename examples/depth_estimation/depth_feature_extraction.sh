#! /bin/bash
cd ../../ #get into the caffe root to run the shell
mkdir examples/_temp
find `pwd`/examples/depth_images -type f -exec echo {} \; > examples/_temp/temp.txt #generate a list of the files to process
sed "s/$/ 0/" examples/_temp/temp.txt > examples/_temp/file_list.txt #add 0 to the end of each line

cp examples/feature_extraction/imagenet_val.prototxt examples/_temp #copy the network definition
./build/tools/extract_features.bin models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel examples/_temp/imagenet_val.prototxt fc7 examples/_temp/features 10 lmdb #everything is in place, build it

