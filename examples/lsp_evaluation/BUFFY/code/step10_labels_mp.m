clear; close all; clc;

startup;

load init.mat;
opts.segmentTypes_mp = {'utor','ltor','ularm','ulelb','urarm','urelb','llelb', ...
    'llarm','lrelb','lrarm','uhead','lhead','lsho','lwai','rsho','rwai', ...
    'mutor','mltor','mularm','murarm','mllarm','mlrarm','mhead','mlsho','mlwai','mrsho','mrwai'};
opts.labels_mp = '../labels/buffy_s5e%d_labels_mp.mat';

save('init.mat','opts');

numpart_new = length(opts.segmentTypes_mp);

p(17,:) = [1 2];
p(18,:) = [1 2];
p(19,:) = [3 4];
p(20,:) = [5 6];
p(21,:) = [7 8];
p(22,:) = [9 10];
p(23,:) = [11 12];
p(24,:) = [13 14];
p(25,:) = [13 14];
p(26,:) = [15 16];
p(27,:) = [15 16];


for e = opts.epi
    fprintf('episode %d/%d\n',e,max(opts.epi));
    load(sprintf(opts.labels,e));
    N = size(labels,3);
    numpart = size(labels,1);
    
    labels_new = zeros(numpart_new,2,N);
    labels_new(1:numpart,:,:) = labels;
    
    for i = [19 20 21 22 23]
        labels_new(i,:,:) = (labels(p(i,1),:,:) + labels(p(i,2),:,:))/2;
    end
    
    for i = [17 24 26]
        labels_new(i,:,:) = labels(p(i,1),:,:)*2/3 + labels(p(i,2),:,:)/3;
    end
    
    for i = [18 25 27]
        labels_new(i,:,:) = labels(p(i,1),:,:)/3 + labels(p(i,2),:,:)*2/3;
    end
    
    labels = labels_new;
    segmentTypes = opts.segmentTypes_mp;
    
    save(sprintf(opts.labels_mp,e),'labels','segmentTypes');
end