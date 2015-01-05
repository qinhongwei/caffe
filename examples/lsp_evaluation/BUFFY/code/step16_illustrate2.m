close all; clear; clc;

startup;

load init.mat;

for e = [2 5 6]
    gt = ReadStickmenAnnotationTxt(sprintf(opts.anno_crop,e));
    
    for n = 1:length(gt)
        frame = gt(n).frame;
        img = imread(sprintf(opts.ims_crop,e,frame));
        figure(1); clf; axis image;
        DrawStickman(gt(n).stickmen.coor, img);
        pause;
    end
    pause;
end