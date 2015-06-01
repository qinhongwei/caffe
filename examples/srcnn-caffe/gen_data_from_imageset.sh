TOOLS=../../build/tools
DATA=/media/cv/Data/srcnn/Cropped-g

$TOOLS/convert_imageset.bin -gray $DATA/data/ $DATA/data/list_train.txt SR-data_lmdb 0 lmdb
$TOOLS/convert_imageset.bin -gray $DATA/label/ $DATA/label/list_train.txt SR-label_lmdb 0 lmdb

$TOOLS/convert_imageset.bin -gray $DATA/data/ $DATA/data/list_test.txt SR-data-test_lmdb 0 lmdb
$TOOLS/convert_imageset.bin -gray $DATA/label/ $DATA/label/list_test.txt SR-label-test_lmdb 0 lmdb
