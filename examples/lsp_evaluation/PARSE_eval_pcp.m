function [detRate PCP R] = PARSE_eval_pcp(name,points,test)

% -------------------
% generate testing stick
I = [1   1   2   2   3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20];
J = [3   15  10  22  10 12 22 24 12 14 24 26 3  5  15 17 5  7  17 19 1  2];
S = [1/2 1/2 1/2 1/2 1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1];
A = full(sparse(I,J,S,20,26));

for n = 1:length(test)
  pts = points{n};
  if isempty(pts)
    predstick(n).stickmen = [];
    continue;
  end    
  for i = 1:size(pts,1)
    predstick(n).stickmen(i).coor = reshape((A*reshape(pts(i,:),2,size(pts,2)/2)')',4,10);
  end
end

% -------------------
% create groundtruth stick
% because PARSE dataset do not have grountruth stick labels, we create the
% groundtruth stick labels by ourselves using groundtruth keypoints
I = [1   1   2   2   3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20];
J = [9   10  3   4   3 2 4 5 2 1 5 6  9  8  10 11 8  7  11 12 14 13];
S = [1/2 1/2 1/2 1/2 1 1 1 1 1 1 1 1  1  1  1  1  1  1  1  1  1  1];
A = full(sparse(I,J,S,20,14));

for n = 1:length(test)
  gtstick(n).stickmen.coor = reshape((A*test(n).obj(1).point(:, 1:2))',4,10);
end

% the PCP evaluation function originally comes from BUFFY dataset, we keep using that for performance evaluation
[detRate PCP R] = eval_pcp('PARSE',predstick,gtstick);
