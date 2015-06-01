mean_mat = caffe('read_mean','/home/cv/image-net/caffe/data/deepfish/deepfish_mean_right.binaryproto');
deepfish_mean = permute(mean_mat,[2,1,3]);
save('deepfish_mean','deepfish_mean');
%mean(abs(new_mat(:)-image_mean(:)))