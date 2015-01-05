close all; clear; clc;

startup;

load init.mat;
opts.groundtruth = '../gtbox/all/buffy_s5e%d_gtbox.mat';

save('init.mat','opts');

for e = opts.epi
    fprintf('episode %d/%d\n',e,max(opts.epi));
    load(sprintf(opts.labels,e));
    load(sprintf(opts.gtbox_face,e));
    
    N = size(labels,3);
    numpart = size(labels,1);
    
    groundtruth = cell(1,numpart);
    for part = 1:numpart
        fprintf('%s\n',opts.segmentTypes{part});
        for n = 1:N
            keypoints = labels(:,:,n);
            facegtbox = gtbox_face(n).box;
            wf = facegtbox(3) - facegtbox(1) + 1;
            hf = facegtbox(4) - facegtbox(2) + 1;
            area = wf*hf;
            boxlen = sqrt(area);
            groundtruth{part}(n,1) = keypoints(part,1) - boxlen/2; 
            groundtruth{part}(n,2) = keypoints(part,2) - boxlen/2;
            groundtruth{part}(n,3) = keypoints(part,1) + boxlen/2;
            groundtruth{part}(n,4) = keypoints(part,2) + boxlen/2;
        end
    end
    save(sprintf(opts.groundtruth,e),'groundtruth');
end