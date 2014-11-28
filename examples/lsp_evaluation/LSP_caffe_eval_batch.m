addpath(genpath('BUFFY'));
clc; clear all; close all;

cache_dir = 'cache';
if ~exist(cache_dir, 'dir')
    mkdir(cache_dir);
end

DEBUG = 0;
TEST_GT = 1;
top_k = 5;
pos_file = '/home/wyang/Code/PE1.41DBN_human_detector/LSP/cache_test_top5/pos.mat';


startiter = 1;
maxiter = 200;
stepsize = 1000;
model_file_idx = (startiter*stepsize:stepsize:maxiter*stepsize);

if TEST_GT
    MODELPATH = '/home/wyang/github/caffe-regression/examples/lsp_alexnet_gt/pose_caffenet_train_iter_';
    PROTOTXT = '/home/wyang/github/caffe-regression/examples/lsp_alexnet_gt/caffenet-pose-lsp-test.prototxt';
else
    MODELPATH = '/home/wyang/github/caffe-regression/examples/lsp_alexnet/pose_caffenet_train_iter_';
    PROTOTXT = '/home/wyang/github/caffe-regression/examples/lsp_alexnet/caffenet-pose-lsp-test.prototxt';
end

LAYER = 'fc8';
LEVELDB = 'train-19-nov-alexnet-prop';
MATDB = [LEVELDB '-mat'];     mkdir(['cache/' MATDB]);
BATCHSIZE = 100;
BATCHNUM=top_k*10;
DIM = 28;

result = [];
if exist(['cache/' MATDB '/result.mat'], 'file')
    load(['cache/' MATDB '/result.mat']);
end
recompute = true;

% % check whether recompute the results
% recompute = false;
% if length(result) ~= length(model_file_idx)
%     for idx = 1:length(model_file_idx)
%         ITER = model_file_idx(idx);
%         caffe_pred = ['cache/' MATDB '/test_labels_' sprintf('%d', ITER) '.mat'];
%         if ~exist(caffe_pred, 'file')
%             recompute = true;
%             continue;
%         end
%     end
% end
% compute results

if recompute
for idx = 1:length(model_file_idx)
    fprintf('Processing #%d | ITER: %d\n', idx, model_file_idx(idx));
    tic;
    ITER = model_file_idx(idx);
    caffe_pred = ['cache/' MATDB '/test_labels_' sprintf('%d', ITER) '.mat'];
    if ~exist(caffe_pred, 'file')
        mycmd = sprintf('sh extract_test_label.sh %s %s %s %s %d %d %d %d %s',...
            MODELPATH, PROTOTXT, LAYER, LEVELDB, ITER, BATCHSIZE, BATCHNUM, DIM, cache_dir);
        system(mycmd);
    end
    %--------------------------------------------
    % Translate predicted pose to original image
    cropsize = 227;
    load(caffe_pred); % predicted

    if ~TEST_GT
        load(pos_file);   % test information
        load '/home/wyang/Code/PE1.41DBN_human_detector/LSP/bbox/lsp_bbox_caffe_0918.mat';  detects = detects(1001:end); % detects bbox
    else

        load('/home/wyang/Code/PE1.41DBN_human_detector/LSP/cache_test_gt/pos.mat');   % test information
        detects = gtbox(1001:end);
    end
    database = '/home/wyang/Datasets/lsp_dataset/';
    imlist = dir([database 'images/*.jpg']);
    imlist = imlist(1001:end);
    
    
    cnt = 0;
    for i = 1:length(imlist) 
%         fprintf('%4d | %4d\n', i, length(imlist));
        [p, name, ext] = fileparts(imlist(i).name);
        im = imread([database 'images/' name ext]);
    %     pim = imread(['/home/wyang/Code/PE1.41DBN_human_detector/LSP/cache_test/pos/' name ext]);
        [h, w, ~] = size(im);
        
        %
        %--------------------------------- 
        % read detection 
        pred = detects(i).pred;
        if isempty(pred)
            pred(1).xmin = 1;
            pred(1).ymin = 1;
            pred(1).xmax = w;
            pred(1).ymax = h;
        end
        
        points{i} = zeros(14, 2);
        for j = 1:min(top_k, length(pred))
            cnt = cnt + 1;
            b = pred(j);
            
%             imshow(im); hold on;
%             plot([b.xmin b.xmin b.xmax b.xmax b.xmin], [b.ymin b.ymax b.ymax b.ymin b.ymin]);
%             pause; close;
            
            [ph, pw, ~] = size(patch(cnt).im);

            prx = feats(cnt, 1:14); 
            pry = feats(cnt, 15:end);
            

            prx = prx*cropsize / (cropsize / pw);
            pry = pry*cropsize / (cropsize / ph);
            

            prx = prx - patch(cnt).lpad + b.xmin;
            pry = pry - patch(cnt).tpad + b.ymin;

            points{i} = points{i} + [prx; pry]'/min(top_k, length(pred));

%             if DEBUG
%                 visualize_pose(im, [prx; pry], ones(1, length(prx)));
%                 pause; close;
%             end
        end
        if DEBUG
            visualize_pose(im, points{i}', ones(1, length(prx)));
            pause; close;
        end
    end

    %-----------------------------------
    % Evaluation
    load('LSP_test_gt_14_pts'); % test
    [detRate, PCP, R] = PARSE_eval_pcp_14(name,points,test);
    if numel(R) == 1
        mr = zeros(1, 6);
    else
        mr = [R(1), 0.5*(R(2)+R(3)), 0.5*(R(4)+R(5)), 0.5*(R(6)+R(7)), 0.5*(R(8)+R(9)), R(10)];
    end
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
    result(idx).detRate = detRate;
    result(idx).mr = mr;
    result(idx).PCP = PCP;  
    result(idx).dPCP = detRate*PCP;  
    result(idx).diff = mdiff;
    toc;
    fprintf('\n-------------------------------------------\n')
end
end
save(['cache/' MATDB '/result.mat'], 'result');


dpcp = zeros(1, length(result));
for i = 1:length(result)
    dpcp(i) = result(i).dPCP;
end
[maxpcp, maxpcpI] = max(dpcp);

fprintf('The model with max PCP is #%d | %d\n', maxpcpI,  (maxpcpI));

detRate = result(maxpcpI).detRate;
mr = result(maxpcpI).mr;
PCP = result(maxpcpI).PCP;
mdiff = result(maxpcpI).diff;

fprintf('PART\ttosal\tU.leg \tL.leg \tU.arm \tL.arm \thead \tTotal \t X diff \t Y diff\n');
fprintf('PCP  '); 
fprintf('\t%.1f ', detRate*mr*100); 
fprintf('\t%.1f', detRate*PCP*100);
fprintf('\t %.2f \t %.2f\n', mdiff(1), mdiff(2));

h = figure;
plot([1:length(model_file_idx)], dpcp);
print(h, '-dpng', [LEVELDB '.png']);



