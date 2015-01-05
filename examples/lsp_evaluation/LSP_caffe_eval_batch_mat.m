addpath(genpath('BUFFY'));
clc; clear all; close all;

DEBUG = 0;
top_k = 5;
cache_prefix = '2014-12-22';

%--------------------------------------------
% Translate predicted pose to original image
cropsize = 227;
caffe_pred = '/home/wyang/github/caffe/examples/lsp_alexnet/extract_features/dec-07-2014-mat/test_labels_fc8_1000.mat';
load(caffe_pred); % predicted
load('/home/wyang/Code/PE1.41DBN_human_detector/LSP/cache_test_top5/pos.mat');   % test information
feats1 = feats(1:length(pos), :);
load '/home/wyang/Code/PE1.41DBN_human_detector/LSP/bbox/lsp_bbox_caffe_0918.mat';  detects = detects(1001:end); % detects bbox


result = [];

itercnt = 0;
for idx = [1000:1000:59000]

    fprintf('Processing ITER: %d\n', idx);
    
load(['/home/wyang/github/caffe/examples/lsp_alexnet/extract_features/dec-22-2014-mat/test_labels_fc9_' int2str(idx) '.mat']);

feats2 = feats(1:length(pos), :);

% feats3 = zeros(length(pos), 4);
% feats3(:, 1) = feats1(:, 13);
% feats3(:, 2) = feats1(:, 14);
% feats3(:, 3) = feats1(:, 27);
% feats3(:, 4) = feats1(:, 28);

% % head
% feats1(:, 13) = feats2(:, 1);
% feats1(:, 14) = feats2(:, 2);
% feats1(:, 27) = feats2(:, 3);
% feats1(:, 28) = feats2(:, 4);

% right arm
feats1(:, 7) = feats2(:, 1);
feats1(:, 8) = feats2(:, 2);
feats1(:, 21) = feats2(:, 3);
feats1(:, 22) = feats2(:, 4);


feats = feats1;


database = '/home/wyang/Datasets/lsp_dataset/';
imlist = dir([database 'images/*.jpg']);
imlist = imlist(1001:end);

cnt = 0;

for i = 1:length(imlist) 
%     fprintf('%4d | %4d\n', i, length(imlist));
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
    
    if DEBUG
        visualize_pose(im, points{i}', ones(1, 14));
        pause; close;
    end
    
    clear cur_points;
    
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
    
    itercnt = itercnt+1;
    result(itercnt).detRate = detRate;
    result(itercnt).mr = mr;
    result(itercnt).PCP = PCP;  
    result(itercnt).dPCP = detRate*PCP;  
    result(itercnt).diff = mdiff;
    fprintf('\n-------------------------------------------\n')
    
end


save(['cache/' cache_prefix '-result.mat'], 'result');

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
plot([1:length(result)], dpcp);
print(h, '-dpng', ['cache' cache_prefix '.png']);
