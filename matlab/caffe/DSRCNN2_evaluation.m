%% evaluate the result of depth_enhancement2, testing the caffemodel:PSNR
clear all;close all;clc;

use_gpu = 1;
caffe('set_device', 2);
model_def_file = '/home/cv/image-net/caffe/models/depth_enhancement2/deploy.prototxt';
model_file = '/home/cv/image-net/caffe/models/depth_enhancement2/snapshots-2x/dsrcnn_iter_2400000.caffemodel';

folderName = 'train'; 
dataPath = ['/media/cv/Data/depth_SR/evaluation/depth_enhancement2/',folderName];
savePath = ['/media/cv/Data/depth_SR/evaluation/depth_enhancement2/hdf5/', folderName, '/'];
fileName = [folderName,'.h5'];
fullfilename = [savePath, fileName];

matcaffe_init(use_gpu, model_def_file, model_file);

%% input files spec

save_path = '/home/cv/image-net/caffe/examples/depth_enhancement2/';
train_path = '/media/cv/Data/depth_SR/evaluation/depth_enhancement2/hdf5/train/';
label_path = '/media/cv/Data/depth_SR/evaluation/depth_enhancement2/hdf5/GT';
%% iterate over the test .mat patches
train_rd = h5read(fullfilename, '/data');
lowdepth_rd = h5read(fullfilename, '/lowDepth');
label_rd = h5read(['/media/cv/Data/depth_SR/evaluation/depth_enhancement2/hdf5/', 'GT/GT.h5'], '/label');
[width, height, dim, count] = size(train_rd);
[width_label, height_label, dim_label, count_label] = size(label_rd);
RMSE = zeros(count,1);
pred = zeros(width_label, height_label, count_label);
for i=1:count
    train_mat = train_rd(:,:,:,i);
    lowdepth_mat = lowdepth_rd(:,:,:,i);
    predict_mat = caffe('forward', {train_mat;lowdepth_mat});
    predict_mat = predict_mat{1};
    pred(:,:,i) = predict_mat;
    label_mat = squeeze(label_rd(:,:,:,i));
    
    %get the MSE
    [m,n] = size(predict_mat);
    diff = (predict_mat - label_mat).^2;
    p = sum(sum(sqrt(diff)))/numel(diff);
    pp = sum(sum(sqrt(lowdepth_mat - label_mat).^2))/numel(label_mat);
    disp(pp);
    RMSE(i) = p;
end

    
    
    
    
    
    