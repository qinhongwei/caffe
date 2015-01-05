close all; clear; clc;

startup;

load init.mat;

% very beautiful data set
% face size follows gaussian distribution with
% mean 74 min 20 max 134 quantile(0.05) 40
% so a 5x5x8 hog template fit it very well

opts.epi = [opts.epi opts.epi+10];

for e = opts.epi
    fprintf('episode %d/%d\n',e,max(opts.epi));
    load(sprintf(opts.gtbox_face,e));
    N(e) = length(gtbox_face);
end

Width = zeros(1,sum(N));
Height = zeros(1,sum(N));

M = 0;
for e = opts.epi
    fprintf('episode %d/%d\n',e,max(opts.epi));
    load(sprintf(opts.gtbox_face,e));
    N(e) = length(gtbox_face);
    
    for n = 1:N(e)
        M = M + 1;
        Width(M) = gtbox_face(n).box(3) - gtbox_face(n).box(1);
        Height(M) = gtbox_face(n).box(4) - gtbox_face(n).box(2);
    end
end

Area = Width .* Height;
Len = sqrt(Area);
hist(Len);