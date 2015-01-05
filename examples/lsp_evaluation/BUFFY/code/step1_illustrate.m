close all; clear; clc;

startup;

load init.mat;

for e = opts.epi
    fprintf('episode %d/%d\n',e,max(opts.epi));
    gt = ReadStickmenAnnotationTxt(sprintf(opts.anno,e));
    N = length(gt);
    load(sprintf(opts.gtbox_face,e));
    for n = 1:N
        fprintf('%d/%d\n',n,N);
        img = imread(sprintf(opts.ims,e,gt(n).frame));
        figure(1); clf; axis image;
        DrawStickman(gt(n).stickmen.coor, img);
        x1 = gtbox_face(n).box(1);
        y1 = gtbox_face(n).box(2);
        x2 = gtbox_face(n).box(3);
        y2 = gtbox_face(n).box(4);
        line([x1 x1 x2 x2 x1]',[y1 y2 y2 y1 y1]','color','g','linewidth',4);
%         pause;
    end
    pause;
end
