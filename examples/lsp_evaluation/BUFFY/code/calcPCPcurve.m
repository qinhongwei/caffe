function [y x] = calcPCPcurve(fun,all_eval_stickmen,all_gt_stickmen,show)
% plotting PCPcurve, 
% input: 
% fun is a function handler to the evaluation routine (e.g. BatchEvalBuffy)
% all_eval_stickmen and all_gt_stickmen as defined in BatchEvalBuffy
% if show is set to true then plotting the curve
% output: [x y] point coordinates on the PCP curve

MIN = 0.1;
STEP = 0.05;
MAX = 0.5;
x = MIN:STEP:MAX;
y = zeros(1,(MAX-MIN)/STEP+1);

idx = 1;
for i=x
  [trash y(idx)] = fun(all_eval_stickmen,all_gt_stickmen,i);
  idx = idx + 1;
end

if show
  plot(x,y);
end


