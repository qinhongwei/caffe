close all; clear; clc;

startup;

load init.mat;
opts.groundtruth_parse = '../gtbox/all/buffy_s5e%d_gtbox_parse.mat';

save('init.mat','opts');

opts.epi = [opts.epi opts.epi+10];

for e = opts.epi
    fprintf('episode %d/%d\n',e,max(opts.epi));
    load(sprintf(opts.labels_parse,e));
    load(sprintf(opts.gtbox_face,e));
    
    N = size(labels,3);
    numpart = size(labels,1);
    
    groundtruth = cell(1,numpart);
    for part = 1:numpart
        fprintf('%s\n',opts.segmentTypes_parse{part});
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
    save(sprintf(opts.groundtruth_parse,e),'groundtruth');
end