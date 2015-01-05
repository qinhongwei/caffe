% get half body labels according to parse structure

close all; clear; clc;

startup;

load init.mat;

opts.labels_parse = '../labels/buffy_s5e%d_labels_parse.mat';

opts.segmentTypes_parse = {'htop','hbot','lsho','lelb','rsho','relb','lwr','rwr','lhip','rhip'};

save('init.mat','opts');

opts.epi = [opts.epi opts.epi+10];

idx_parse{1} =  11;
idx_parse{2} = [1 12];
idx_parse{3} = 3;
idx_parse{4} = [4 7];
idx_parse{5} = 5;
idx_parse{6} = [6 9];
idx_parse{7} = 8;
idx_parse{8} = 10;
idx_parse{9} = 14;
idx_parse{10} = 16;

for e = opts.epi
    fprintf('episode %d/%d\n',e,max(opts.epi));
    load(sprintf(opts.labels,e));
    
    N = size(labels,3);
    P = length(idx_parse);
    labels_new = zeros(P,2,N);
    for n = 1:N
        for i = 1:P
            labels_new(i,:,n) = mean(labels(idx_parse{i},:,n),1);
        end
    end
    labels = labels_new;
    segmentTypes = opts.segmentTypes_parse;
    save(sprintf(opts.labels_parse,e),'labels','segmentTypes');
end