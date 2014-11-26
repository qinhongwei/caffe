close all; clear; clc;

startup;

opts.epi = [2 3 4 5 6];
opts.segmentTypes = {'torsso','luarm','ruarm','llarm','rlarm','face'};
opts.anno = '../data/buffy_s5e%d_sticks.txt';
opts.ims = '../images/buffy_s5e%d_original/%.6d.jpg';
opts.gtbox_face = '../gtbox/face/buffy_s5e%d_box.mat';

save('init.mat','opts');

for e = opts.epi
    fprintf('episode %d/%d\n',e,max(opts.epi));
    gt = ReadStickmenAnnotationTxt(sprintf(opts.anno,e));
    N = length(gt);
    curlen = 0;
    clear gtbox_face;
    if exist(sprintf(opts.gtbox_face,e),'file')
        load(sprintf(opts.gtbox_face,e));
        curlen = length(gtbox_face);
    end
    for n = 1:N
        if n <= curlen;
            continue;
        end
        fprintf('%d/%d\n',n,N);
        img = imread(sprintf(opts.ims,e,gt(n).frame));
        figure(1); clf; axis image;
        DrawStickman(gt(n).stickmen.coor, img);
        [x y] = ginput(2);
        % [x y] = getrect;
        gtbox_face(n).frame = gt(n).frame;
        gtbox_face(n).box(1) = x(1);
        gtbox_face(n).box(2) = y(1);
        gtbox_face(n).box(3) = x(2);
        gtbox_face(n).box(4) = y(2);
        figure(2); clf; showboxes(img,gtbox_face(n).box);
        
        save(sprintf(opts.gtbox_face,e),'gtbox_face');
    end
    save(sprintf(opts.gtbox_face,e),'gtbox_face');
end