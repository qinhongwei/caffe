close all; clear; clc;

load init.mat;

for e = opts.epi
    fprintf('episode %d/%d\n',e,max(opts.epi));
    gt = ReadStickmenAnnotationTxt(sprintf(opts.anno,e));
    load(sprintf(opts.labels,e));
    
    N = length(gt);
    for n = 1:N
        fprintf('%d/%d\n',n,N);
        img = imread(sprintf(opts.ims,e,gt(n).frame));
        figure(1); clf; axis image;
        tmplabels = labels(:,:,n);
        numpart = size(labels,1);
        tmplabels = tmplabels';
        DrawStickman(reshape(tmplabels,4,numpart/2), img);
    end
    pause;
end
