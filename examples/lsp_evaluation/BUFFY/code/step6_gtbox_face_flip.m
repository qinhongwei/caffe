close all; clear; clc;

startup;

load init.mat;

for e = opts.epi
    fprintf('episode %d/%d\n',e,max(opts.epi));
    load(sprintf(opts.gtbox_face,e));
    N = length(gtbox_face);
    
    clear gtbox_facenew;
    for n = 1:N
        im = imread(sprintf(opts.ims,e,gtbox_face(n).frame));
        Width = size(im,2);
        
        box = gtbox_face(n).box;
        gtbox_facenew(n).box(1) = Width - box(3) + 1;
        gtbox_facenew(n).box(2) = box(2);
        gtbox_facenew(n).box(3) = Width - box(1) + 1;
        gtbox_facenew(n).box(4) = box(4);        
        gtbox_facenew(n).frame = gtbox_face(n).frame;
    end
    gtbox_face = gtbox_facenew;
    save(sprintf(opts.gtbox_face,e+10),'gtbox_face');    
end