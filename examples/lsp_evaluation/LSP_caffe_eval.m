addpath(genpath('BUFFY'));
clc; clear all; close all;

DEBUG = 1;
top_k = 5;

%--------------------------------------------
% Translate predicted pose to original image
cropsize = 227;
caffe_pred = '../lsp_elw/extract_feature/train-14-nov/test_labels.mat';
load(caffe_pred); % predicted
load('/home/wyang/Code/PE1.41DBN_human_detector/LSP/cache_test_top5/pos.mat');   % test information
feats = feats(1:length(pos), :);
load '/home/wyang/Code/PE1.41DBN_human_detector/LSP/bbox/lsp_bbox_caffe_0918.mat';  detects = detects(1001:end); % detects bbox

database = '/home/wyang/Datasets/lsp_dataset/';
imlist = dir([database 'images/*.jpg']);
imlist = imlist(1001:end);

cnt = 0;

for i = 1:length(imlist) 
    fprintf('%4d | %4d\n', i, length(imlist));
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
        b = pred(j);
        cnt = cnt + 1;
        [ph, pw, ~] = size(patch(cnt).im);
        prx = feats(cnt, 1:14); 
        pry = feats(cnt, 15:end);

        prx = prx*cropsize / (cropsize / pw);
        pry = pry*cropsize / (cropsize / ph);

        prx = prx - patch(cnt).lpad + b.xmin;
        pry = pry - patch(cnt).tpad + b.ymin;
        
        points{i} = points{i} + [prx; pry]'/min(top_k, length(pred));
    
        if DEBUG
            visualize_pose(im, [prx; pry], ones(1, length(prx)));
            pause; close;
        end
    end   
    
%     if DEBUG
%         visualize_pose(im, points{i}', ones(1, 14));
%         pause; close;
%     end
    
    clear cur_points;
    
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
