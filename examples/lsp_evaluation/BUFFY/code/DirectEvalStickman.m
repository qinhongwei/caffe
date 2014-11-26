function [Score R] = DirectEvalStickman(estimated_stickman,groundtruth_stickman, pcp_matching_threshold)
% user interface to evaluate single stickman against a single ground-truth stickman
% estimated_stickman and groundtruth_stickman are arrays of stickman end-points coordinates (:,nparts) = [x1 y1 x2 y2]'
% pcp_matching_threshold (optional) - defines the PCP sticks matching threshold (default 0.5) -> for definition look into README.txt 
%
% Output:
% Score - number of correctly estimated body parts 
% R(p) - indicator of parts that have been correctly estimated
%
% Eichner/2009

if nargin < 3
  pcp_matching_threshold = 0.5;
end

% evaluation criterion used in Ferrari (cvpr08, cvpr09) and Eichner (bmvc09) papers
pars.max_parall_dist = pcp_matching_threshold;
pars.max_perp_dist = pcp_matching_threshold;

R = DirectEvalSegms(double(estimated_stickman),double(groundtruth_stickman),pars);
Score = sum(R);