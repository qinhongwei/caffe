close all; clear; clc;

startup;

load init.mat;

colors = sampleColors();

for e = opts.epi
    fprintf('episode %d/%d\n',e,max(opts.epi));
    gt = ReadStickmenAnnotationTxt(sprintf(opts.anno,e));
    N = length(gt);
    
    for n = 1:N
        fprintf('%d/%d\n',n,N);
        img = imread(sprintf(opts.ims,e,gt(n).frame));
        figure(1); clf; axis image;
        imshow(img); hold on;
        Coor = gt(n).stickmen.coor;
        Coor = reshape(Coor,2,size(Coor,2)*2);
        for p = 1:size(Coor,2)
            plot(Coor(1,p),Coor(2,p),'o','markersize',10,'color',colors(p,:),'markerfacecolor',colors(p,:));
        end
        pause;
    end
    pause;
end
