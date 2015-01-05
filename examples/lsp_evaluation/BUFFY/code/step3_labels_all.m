close all; clear; clc;

startup;

load init.mat;
opts.labels = '../labels/buffy_s5e%d_labels.mat';
opts.segmentTypes = {'utor','ltor','ularm','ulelb','urarm','urelb','llelb','llarm','lrelb','lrarm','uhead','lhead','lsho','lwai','rsho','rwai'};
segmentTypes = opts.segmentTypes;

save('init.mat','opts');

for e = opts.epi
    fprintf('episode %d/%d\n',e,max(opts.epi));
    gt = ReadStickmenAnnotationTxt(sprintf(opts.anno,e));
    load(sprintf(opts.labels_extra,e));
    N = length(gt);
    
    len1 = size(gt(1).stickmen.coor,2)*2; 
    len2 = size(labels_extra(:,:,1),1);
    len = len1 + len2;
    labels = zeros(len,2,N);
    for n = 1:N
        fprintf('%d/%d\n',n,N);
        labels(1:len1,:,n) = reshape(gt(n).stickmen.coor,2,len1)';
        labels(len1+1:end,:,n) = labels_extra(:,:,n);
    end
    
    save(sprintf(opts.labels,e),'labels','segmentTypes');
end