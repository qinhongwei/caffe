%% evaluate the result of depth_enhancement, testing the caffemodel:PSNR
clear all;close all;clc;

use_gpu = 1;
caffe('set_device', 0);
model_def_file = '/home/cv/image-net/caffe/models/depth_enhancement/deploy.prototxt';
model_file = '/home/cv/image-net/caffe/models/depth_enhancement/snapshots-2x-finetune/dsrcnn_iter_2400000.caffemodel';

folderName = 'train'; 
dataPath = ['/media/cv/Data/depth_SR/evaluation/',folderName];
savePath = ['/media/cv/Data/depth_SR/evaluation/hdf5/', folderName, '/'];
fileName = [folderName,'.h5'];
fullfilename = [savePath, fileName];

matcaffe_init(use_gpu, model_def_file, model_file);

%% input files spec

save_path = '/home/cv/image-net/caffe/examples/depth_enhancement/';
train_path = '/media/cv/Data/depth_SR/evaluation/hdf5/train/';
label_path = '/media/cv/Data/depth_SR/evaluation/hdf5/GT';
%% iterate over the test .mat patches
train_rd = h5read(fullfilename, '/data');
label_rd = h5read(['/media/cv/Data/depth_SR/evaluation/hdf5/', 'GT/GT.h5'], '/label');
[width, height, dim, count] = size(train_rd);
MSE = zeros(100,1);
pred = zeros(20, 20, count);
for i=1:count
    train_mat = train_rd(:,:,:,i);
    predict_mat = caffe('forward', {train_mat});
    predict_mat = predict_mat{1};
    pred(:,:,i) = predict_mat;
    label_mat = squeeze(label_rd(:,:,:,i));
    
    %get the MSE
    [m,n] = size(predict_mat);
    diff = (predict_mat - label_mat).^2;
    p = sum(sum(diff))/(2*m*n);
    MSE(i) = p;
end

    
    
    
    
    
    