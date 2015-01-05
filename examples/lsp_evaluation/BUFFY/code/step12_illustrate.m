close all; clear; clc;

load init.mat;

for e = opts.epi
    fprintf('episode %d/%d\n',e,max(opts.epi));
    load(sprintf(opts.groundtruth_mp,e+10));
    load(sprintf(opts.labels_mp,e+10));
    load(sprintf(opts.gtbox_face,e+10));
    N = size(labels,3);
    numpart = size(labels,1);
    for n = 1:N
        fprintf('%d/%d\n',n,N);
        for part = 1:numpart
            box(part,:) = groundtruth{part}(n,:);
        end        
        img = imread(sprintf(opts.ims,e+10,gtbox_face(n).frame));
        figure(1); clf; axis image;
        showboxes(img,box);
        pause;
    end
    pause;
end
