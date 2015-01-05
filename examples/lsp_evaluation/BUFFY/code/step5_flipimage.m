close all; clear; clc;

startup;

load init.mat;

for e = opts.epi
    fprintf('episode %d/%d\n',e,max(opts.epi));
    gt = ReadStickmenAnnotationTxt(sprintf(opts.anno,e));
    N = length(gt);
    for n = 1:N
        im = imread(sprintf(opts.ims,e,gt(n).frame));
        imflip = im(:,end:-1:1,:);
        imwrite(imflip,sprintf(opts.ims,e+10,gt(n).frame));
    end
end