"""
This script generates data for SRCNN, X. Tang, ECCV2014

The data is generated by randomly cropping into 32*32 sub-images.
Label is generated by performing conjugate resizing in size k and 1/k.
"""

import cv2
import os
import random
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Parameters
window_w = 32
window_h = 32
sample_per_image = 1000

zooming = 2

test_ratio = 0.1

data_path = '/media/cv/Data/srcnn/Training/'
output_path = '/media/cv/Data/srcnn/Cropped-g/'

# erase previous run
def clear_folder(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception, e:
            print e
clear_folder(output_path + 'data')
clear_folder(output_path + 'label')



# List file
file_list = []
list_data_train  = open(output_path + 'data/list_train.txt', 'w')
list_label_train = open(output_path + 'label/list_train.txt', 'w')
list_data_test = open(output_path + 'data/list_test.txt', 'w')
list_label_test = open(output_path + 'label/list_test.txt', 'w')

# Mean file
mean = np.zeros((window_h, window_w))
cnt = 0
for fn in os.listdir(data_path):
    mat = cv2.imread(data_path + fn)
    ycrcb = cv2.cvtColor(mat, cv2.COLOR_RGB2YCR_CB)
    mat = ycrcb[:,:,0]
    for x0 in range(0, mat.shape[1] - window_w + 1, 14):  #stride 14
        for y0 in range(0, mat.shape[0] - window_h + 1, 14):
            data = mat[y0:y0+window_h, x0:x0+window_w]
            label = data[6:-6, 6:-6] # 20*20
            
            # low-resolution
            data_small = cv2.resize(data, (int(data.shape[0]/zooming), int(data.shape[1]/zooming)))  #ignore the method of interpolation
            data = cv2.resize(data_small, (data.shape[0], data.shape[1]), interpolation=cv2.INTER_CUBIC)
            
            out_fn = '%s-x%d-y%d.bmp' % (fn[0:-4], x0, y0)
            # write data image
            cv2.imwrite('%sdata/%s' % (output_path, out_fn), data)
            # write label imgae
            cv2.imwrite('%slabel/%s' % (output_path, out_fn), label)
            
            # mean
            mean = mean + data
            cnt = cnt + 1
            
            # file name list
            file_list.append(out_fn)

# Mean
mean = mean / cnt
mean_file = open(output_path + 'data/image_mean.npy', 'wb')
pickle.dump(mean, mean_file)
file_list = np.random.permutation(file_list)

# write list file
#split = int(len(file_list) * (1-test_ratio))
split = 2544
for i in range(cnt-split):
    list_data_train.write(file_list[i]  + ' 0\n')
    list_label_train.write(file_list[i] + ' 0\n')
for i in range(cnt-split, len(file_list)):
    list_data_test.write(file_list[i]   + ' 0\n')
    list_label_test.write(file_list[i] + ' 0\n')

# List
list_data_train.close()
list_label_train.close()
list_data_test.close()
list_label_test.close()

print 'split '+str(split)+'\n'
print 'all '+str(cnt)+'\n'
print 'Done!'
