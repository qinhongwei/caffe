#!/usr/bin/env python

import leveldb
import binascii
import numpy as np
import sys
sys.path.insert(0,'/home/cv/image-net/caffe/python')
import caffe
from caffe.proto import caffe_pb2

# parse input argument
dbName = '/home/cv/image-net/caffe/examples/_temp/features'

# open leveldb files

db = leveldb.LevelDB(dbName)

# get db iterator

it = db.RangeIter()
print 'success 0'

for key,value in it:
# convert string to datum
    print 'success 1'
    datum = caffe_pb2.Datum.FromString(db.Get(key))
    arr = caffe.io.datum_to_array(datum)[0]
    i=0
    tmpS=''
    for i in range(0,len(arr)):
        tmpS+=str(i+1)+':'+str(arr[i].tolist()[0])+''
    print 'success 2'
    print tmpS

