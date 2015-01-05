clc; close all;
name = 'LSP'
%------------------------------------
% EVAL 26 points LSP
fprintf('\n+++++++++++++++++++++++++ CVPR 2014   +++++++++++++++++++++++\n');
fprintf('\n------------------------- 26 Points   -----------------------\n');
addpath(genpath('BUFFY/code'));
load('results/wanli_cvpr14_26.mat');
load /home/wyang/Datasets/lsp_dataset/LSP_gt_14_pts.mat
test = test(1001:2000);
[detRate, PCP, R] = PARSE_eval_pcp(name,points2,test);
mr = [R(1), 0.5*(R(2)+R(3)), 0.5*(R(4)+R(5)), 0.5*(R(6)+R(7)), 0.5*(R(8)+R(9)), R(10)];
fprintf('Strict detRate=%.3f, PCP=%.3f, detRate*PCP=%.3f\n',detRate,PCP,detRate*PCP);
fprintf('PART,\ttosal,\tU.leg, \tL.leg, \tU.arm, \tL.arm, \thead, \tTotal\n');
fprintf('PCP  '); 
fprintf(',\t%.1f ', detRate*mr*100); 
fprintf(',\t%.1f\n', detRate*PCP*100);

%------------------------------------
% EVAL 14 points LSP
fprintf('\n------------------------- 14 Points   -----------------------\n');
load('results/wanli_cvpr14_14.mat');
[detRate, PCP, R] = PARSE_eval_pcp_14(name,points,test);
mr = [R(1), 0.5*(R(2)+R(3)), 0.5*(R(4)+R(5)), 0.5*(R(6)+R(7)), 0.5*(R(8)+R(9)), R(10)];
fprintf('Strict detRate=%.3f, PCP=%.3f, detRate*PCP=%.3f\n',detRate,PCP,detRate*PCP);
fprintf('PART,\ttosal,\tU.leg, \tL.leg, \tU.arm, \tL.arm, \thead, \tTotal\n');
fprintf('PCP  '); 
fprintf(',\t%.1f ', detRate*mr*100); 
fprintf(',\t%.1f\n', detRate*PCP*100);

diff = [];
for i = 1:length(points)
    pred = points{i};
    grnd = test(i).point;
    diff = [diff; abs((pred - grnd))];
end
mdiff = mean(diff);
fprintf('mean X diff: %.2f, mean Y diff: %.2f\n', mdiff(1), mdiff(2));

%------------------------------------
% EVAL Testing results
% 26 points LSP
fprintf('\n+++++++++++++++++++++++++ TESTING    +++++++++++++++++++++++\n');
fprintf('\n------------------------- 26 Points   -----------------------\n');
load('results/lsp_detections_0901.mat');
proposals = dets(1001:2000);
pts = cell(1, 1000);
for i = 1:length(proposals)
   pts{i} = reshape(proposals(i).obj(1).point', 1, 52); 
end

[detRate, PCP, R] = PARSE_eval_pcp(name,pts,test);
mr = [R(1), 0.5*(R(2)+R(3)), 0.5*(R(4)+R(5)), 0.5*(R(6)+R(7)), 0.5*(R(8)+R(9)), R(10)];
fprintf('Strict detRate=%.3f, PCP=%.3f, detRate*PCP=%.3f\n',detRate,PCP,detRate*PCP);
fprintf('PART,\ttosal,\tU.leg, \tL.leg, \tU.arm, \tL.arm, \thead, \tTotal\n');
fprintf('PCP  '); 
fprintf(',\t%.1f ', detRate*mr*100); 
fprintf(',\t%.1f\n', detRate*PCP*100);

%------------------------------------
% EVAL Testing results
% 14 points LSP
fprintf('\n------------------------- 14 Points   -----------------------\n');
I = [1  2  3  4  5  6  7  8  9  10 11 12 13 14];
J = [14 12 10 22 24 26 7  5  3  15 17 19 2  1];
A = [1  1  1  1  1  1  1  1  1  1  1  1  1  1];
Transback = full(sparse(I,J,A,14,26));
pts = cell(1, 1000);
for i = 1:length(proposals)
    pts{i} = Transback*proposals(i).obj(1).point;
end

% database = '/home/wyang/Datasets/lsp_dataset/';
% imlist = dir([database 'images/*.jpg']);
% imlist = imlist(1001:end);
% for i = 1:length(imlist) 
%     fprintf('%4d | %4d\n', i, length(imlist));
%     [p, name, ext] = fileparts(imlist(i).name);
%     
%     im = imread([database 'images/' name ext]);
%     visualize_pose(im, pts{i}', ones(1, length(prx)));
%     pause; close;
%     
% end

[detRate, PCP, R] = PARSE_eval_pcp_14(name,pts,test);
mr = [R(1), 0.5*(R(2)+R(3)), 0.5*(R(4)+R(5)), 0.5*(R(6)+R(7)), 0.5*(R(8)+R(9)), R(10)];
fprintf('Strict detRate=%.3f, PCP=%.3f, detRate*PCP=%.3f\n',detRate,PCP,detRate*PCP);
fprintf('PART,\ttosal,\tU.leg, \tL.leg, \tU.arm, \tL.arm, \thead, \tTotal\n');
fprintf('PCP  '); 
fprintf(',\t%.1f ', detRate*mr*100); 
fprintf(',\t%.1f\n', detRate*PCP*100);


