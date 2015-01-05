close all; clear; clc;

startup;

load init.mat;

save('init.mat','opts');

m(1) = 1; m(2) = 2; m(3) = 5; m(4) = 6; m(5) = 3; m(6) = 4;
m(7) = 9; m(8) = 10; m(9) = 7; m(10) = 8; m(11) = 11; m(12) = 12;
m(13) = 15; m(14) = 16; m(15) = 13; m(16) = 14;
m(17) = 17; m(18) = 18; m(19) = 20; m(20) = 19; m(21) = 22; m(22) = 21;
m(23) = 23; m(24) = 26; m(25) = 27; m(26) = 24; m(27) = 25;

for e = opts.epi
    fprintf('episode %d/%d\n',e,max(opts.epi));
    gt = ReadStickmenAnnotationTxt(sprintf(opts.anno,e));
    load(sprintf(opts.labels_mp,e));
    
    N = size(labels,3);
    P = size(labels,1);
    labels_new = zeros(P,2,N);
    for n = 1:N
        im = imread(sprintf(opts.ims,e,gt(n).frame));
        Width = size(im,2);
        points = labels(:,:,n);
        
        for i = 1:P
            labels_new(m(i),1,n) = Width - points(i,1) + 1;
            labels_new(m(i),2,n) = points(i,2);
        end
    end
    labels = labels_new;
    save(sprintf(opts.labels_mp,e+10),'labels','segmentTypes');
end