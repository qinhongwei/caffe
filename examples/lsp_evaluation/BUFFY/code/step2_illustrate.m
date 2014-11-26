close all; clear; clc;

load init.mat;

for e = opts.epi
    fprintf('episode %d/%d\n',e,max(opts.epi));
    gt = ReadStickmenAnnotationTxt(sprintf(opts.anno,e));
    load(sprintf(opts.labels_extra,e));
    
    N = length(gt);
    for n = 1:N
        fprintf('%d/%d\n',n,N);
        img = imread(sprintf(opts.ims,e,gt(n).frame));
        figure(1); clf; axis image;
        DrawStickman(gt(n).stickmen.coor, img);
        
        x1 = labels_extra(1,1,n);
        y1 = labels_extra(1,2,n);
        x2 = labels_extra(2,1,n);
        y2 = labels_extra(2,2,n);
        line([x1 x2]',[y1 y2]','color','w','linewidth',4);
        
        x1 = labels_extra(3,1,n);
        y1 = labels_extra(3,2,n);
        x2 = labels_extra(4,1,n);
        y2 = labels_extra(4,2,n);
        line([x1 x2]',[y1 y2]','color','w','linewidth',4);
    end
    pause;
end
