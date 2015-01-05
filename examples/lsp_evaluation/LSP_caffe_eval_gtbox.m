addpath(genpath('BUFFY'));
clc; clear all; close all;

cache_dir = 'cache';
if ~exist(cache_dir, 'dir')
    mkdir(cache_dir);
end

DEBUG = 0;
startiter = 1;
maxiter = 162;
stepsize = 1000;
model_file_idx = (startiter*stepsize:stepsize:maxiter*stepsize);


MODELPATH = '/home/wyang/github/caffe-regression/examples/lsp_elw/cache/train-07-nov/train-07-nov-finetune-from-alexnet/pose_caffenet_train_iter_';
ITER = 145000;
PROTOTXT = '/home/wyang/github/caffe-regression/examples/lsp_elw/cache/train-07-nov/train-07-nov-finetune-from-alexnet/caffenet-pose-lsp-test-gt.prototxt';
LAYER = 'elw10';
LEVELDB = 'train-06-nov-gt';
MATDB = [LEVELDB '-mat'];     mkdir(['cache/' MATDB]);
BATCHSIZE = 100;
BATCHNUM=10;
DIM = 28;

caffe_pred = ['cache/' MATDB '/test_labels_' sprintf('%d', ITER) '.mat'];

mycmd = sprintf('sh extract_test_label.sh %s %s %s %s %d %d %d %d %s',...
    MODELPATH, PROTOTXT, LAYER, LEVELDB, ITER, BATCHSIZE, BATCHNUM, DIM, cache_dir);
system(mycmd);
%--------------------------------------------
% Translate predicted pose to original image
cropsize = 227;
load(caffe_pred); % predicted


load('/home/wyang/Code/PE1.41DBN_human_detector/LSP/cache_test_gt/pos.mat');   % test information
load '/home/wyang/Code/PE1.41DBN_human_detector/LSP/bbox/lsp_bbox_caffe_0918.mat';  detects = detects(1001:end); % detects bbox

database = '/home/wyang/Datasets/lsp_dataset/';
imlist = dir([database 'images/*.jpg']);
imlist = imlist(1001:end);
for i = 1:length(imlist) 
%         fprintf('%4d | %4d\n', i, length(imlist));
    [p, name, ext] = fileparts(imlist(i).name);
    im = imread([database 'images/' name ext]);
%     pim = imread(['/home/wyang/Code/PE1.41DBN_human_detector/LSP/cache_test/pos/' name ext]);
    [h, w, ~] = size(im);
    [ph, pw, ~] = size(patch(i).im);

    % bounding box

     b = pos(i).b;

    prx = feats(i, 1:14); 
    pry = feats(i, 15:end);

    [sh, sw, ~] = size(patch(i).im);
    scale = sh / cropsize;
    prx = prx*cropsize / (cropsize / pw);
    pry = pry*cropsize / (cropsize / ph);

    prx = prx - patch(i).lpad + b.xmin;
    pry = pry - patch(i).tpad + b.ymin;

    points{i} = [prx; pry]';
    if DEBUG
        visualize_pose(im, [prx; pry], ones(1, length(prx)));
        pause; close;
    end
end

%-----------------------------------
% Evaluation
load('LSP_test_gt_14_pts'); % test
[detRate, PCP, R] = PARSE_eval_pcp_14(name,points,test);
mr = [R(1), 0.5*(R(2)+R(3)), 0.5*(R(4)+R(5)), 0.5*(R(6)+R(7)), 0.5*(R(8)+R(9)), R(10)];
fprintf('Strict detRate=%.3f, PCP=%.3f, detRate*PCP=%.3f\n',detRate,PCP,detRate*PCP);
fprintf('PART\ttosal\tU.leg \tL.leg \tU.arm \tL.arm \thead \tTotal \t X diff \t Y diff\n');
fprintf('PCP  '); 
fprintf('\t%.1f ', detRate*mr*100); 
fprintf('\t%.1f', detRate*PCP*100);diff = [];

diff = [];
for i = 1:length(points)
    pred = points{i};
    grnd = test(i).obj(1).point;
    diff = [diff; abs((pred - grnd))];
end
mdiff = mean(diff);
fprintf('\t %.2f \t %.2f\n', mdiff(1), mdiff(2));
toc;
fprintf('\n-------------------------------------------\n')
