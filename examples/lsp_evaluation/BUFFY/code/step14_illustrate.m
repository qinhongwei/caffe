close all; clear; clc;

load init.mat;

for e = opts.epi
    fprintf('episode %d/%d\n',e,max(opts.epi));
    load(sprintf(opts.groundtruth_parse,e));
    load(sprintf(opts.labels_parse,e));
    load(sprintf(opts.gtbox_face,e));
    N = size(labels,3);
    numpart = size(labels,1);
    for n = 1:N
        fprintf('%d/%d\n',n,N);
        for part = 1:numpart
            box(part,:) = groundtruth{part}(n,:);
        end        
        img = imread(sprintf(opts.ims,e,gtbox_face(n).frame));
        figure(1); clf; axis image;
        showboxes(img,box);
        pause;
    end
    pause;
end
