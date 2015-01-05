function [detRate PCP R] = BatchEvalBuffy( all_eval_stickmen, all_gt_stickmen, pcp_matching_threshold)
% user interface to evaluate stickmen sets for more that one frame 
% (multi-frame evaluation)
%
% Input:
% all_eval_stickmen - (contains stickmen to be evaluated for multiple frame) is a struct arrays (one entry per frames) with the following fields:
%                   .episode - corresponding episode number
%                   .frame - corresponding frame number
%                   .stickmen - struct array containing all stickmen for this frame, with fields:
%                       .coor - stickman end-points coordinates coor(:,nparts) = [x1 y1 x2 y2]'
%                       .det - is the detection window associated with the stickman in [minx miny maxx maxy]
% 
%                       if .det field doesn't exist, then a detection window will be derived automatically from head and torso sticks
%                       for an example look at BMVC09best256.mat
%
% all_gt_stickman - (contains ground-truth annotations for multiple frames) is a struct array of ground-truth stickmen (one entry per frame) with fields:
%                   .episode - corresponding episode number
%                   .frame - corresponding frame number     
%                   .stickmen - 1x1 struct containing ground-truth stickmen:       
%                       .coor - stickman end-points coordinates (:,nparts) = [x1 y1 x2 y2]'
%
% pcp_matching_threshold (optional) - defines the PCP sticks matching threshold (default 0.5) -> for definition look into README.txt 
%
% number of elements in all_gt_stickman and all_eval_stickmen must be the same 
% because every element in all_eval_stickmen is evaluated against a corresponding element in all_gt_stickman
% (but the order of these elements doesn't matter)
%
% Output:
% detRate - is the detection rate of the system (see README.txt)
% PCP - Percentage of Correctly estimated body Parts, evaluated only in correct detections (see README.txt)
%
% see also EvalStickmen.m 
%
% Eichner/2009

if nargin < 3
  pcp_matching_threshold  = 0.5;
end

nLimbs = size(all_gt_stickmen(1).stickmen.coor,2);

total = length(all_eval_stickmen);

assert( total == length(all_gt_stickmen))

scores = zeros(total,1);


% make sure that all_gt_stickmen and all_eval_stickmen are in the same order (wrt to frames and episodes)
gtframes = [all_gt_stickmen(:).frame];
evframe = [all_eval_stickmen(:).frame];

[trash idx] = sort(gtframes);
all_gt_stickmen = all_gt_stickmen(idx);
[trash idx] = sort(evframe);
all_eval_stickmen = all_eval_stickmen(idx);

gtepisode = [all_gt_stickmen(:).episode];
evepisode = [all_eval_stickmen(:).episode];

[trash idx] = sort(gtepisode);
all_gt_stickmen = all_gt_stickmen(idx);
[trash idx] = sort(evepisode);
all_eval_stickmen = all_eval_stickmen(idx);


% evaluate each frame
for i=1:total
  if all_eval_stickmen(i).frame ~= all_gt_stickmen(i).frame || all_eval_stickmen(i).episode ~= all_gt_stickmen(i).episode
    error('set given for evaluation does not correspond to the ground-truth');
  end
  [trash scores(i) Rs(i,:)] = EvalStickmen(all_eval_stickmen(i).stickmen, all_gt_stickmen(i).stickmen, pcp_matching_threshold);
end


% compute detRate and PCP
matched = ~isnan(scores);
nMatched = sum(matched);
detRate = nMatched/total;
PCP = sum(scores(matched))/(nMatched*nLimbs);
R = sum(Rs(matched,:))/(nMatched);

end

