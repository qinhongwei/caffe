close all; clear; clc;

startup;

load init.mat;

opts.labels_extra = '../labels/buffy_s5e%d_labels_extra.mat';

save('init.mat','opts');

for e = opts.epi
    fprintf('episode %d/%d\n',e,max(opts.epi));
    gt = ReadStickmenAnnotationTxt(sprintf(opts.anno,e));
    N = length(gt);
    
    curlen = 0;
    clear labels_extra;
    if exist(sprintf(opts.labels_extra,e),'file')
        load(sprintf(opts.labels_extra,e));
        curlen = length(labels_extra);
    end
    for n = 1:N
        if n <= curlen;
            continue;
        end
        fprintf('%d/%d\n',n,N);
        img = imread(sprintf(opts.ims,e,gt(n).frame));
        figure(1); clf; axis image;
        DrawStickman(gt(n).stickmen.coor, img);
        [x y] = ginput(4);
        labels_extra(:,:,n) = [x y];
        
        save(sprintf(opts.labels_extra,e),'labels_extra');
    end
    save(sprintf(opts.labels_extra,e),'labels_extra');
end