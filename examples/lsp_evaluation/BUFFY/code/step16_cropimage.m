close all; clear; clc;

startup;

load init.mat;
opts.ims_crop = '../images/buffy_s5e%d_crop/%.6d.jpg';
opts.anno_crop = '../data/buffy_s5e%d_sticks_crop.txt';
save('init.mat','opts');

load ../BMVC09best256.mat;

for n = 1:length(BMVC09best256)
    e(n) = BMVC09best256(n).episode;
    frame(n) = BMVC09best256(n).frame;
    idx(n) = 0;
    fprintf('%d/%d\n',e(n),frame(n));
    gt = ReadStickmenAnnotationTxt(sprintf(opts.anno,e(n)));
    img = imread(sprintf(opts.ims,e(n),frame(n)));
    
    if isempty(BMVC09best256(n).stickmen)
        continue;
    end
    stick = BMVC09best256(n).stickmen;
    m = 0;
    for i = 1:length(gt)
        if gt(i).frame == frame(n)
            m = i;
            break;
        end
    end
    [evalNo, Score] = EvalStickmen(stick, gt(m).stickmen, 0.5);
    
    if evalNo 
        idx(n) = 1;
    end
end

for e = [2 5 6]
    M = 0;
    gt = ReadStickmenAnnotationTxt(sprintf(opts.anno,e));
    clear gt_modify;
    clear testframe;
    for n = 1:length(BMVC09best256)
        epi = BMVC09best256(n).episode;
        if epi ~= e
            continue;
        end
        if idx(n) == 0
            continue;
        end
        
        
        M = M + 1;
        frame = BMVC09best256(n).frame;
        stick = BMVC09best256(n).stickmen;
        img = imread(sprintf(opts.ims,e,frame));
        m = 0;
        for i = 1:length(gt)
            if gt(i).frame == frame
                m = i;
                break;
            end
        end
        fprintf('%d/%d\n',e,frame);
        
        pars.det_hwratio(2) = 0.9;
        pars.iou_thresh = 0.5;
        gt_bb = detBBFromStickman(gt(m).stickmen.coor,'ubf',pars);
        
        box = gt_bb;
        x1 = box(1);
        y1 = box(2);
        x2 = box(3);
        y2 = box(4);
        Width = x2 - x1; Height = y2 - y1;
        x1 = round(x1 - Width*8/10);
        y1 = round(y1 - Height*2/3);
        x2 = round(x2 + Width*8/10);
        y2 = round(y2 + Height*13/10);
        
        [H W dummy] = size(img);
        im = img(max(1,y1):min(H,y2),max(1,x1):min(W,x2),:);
        imwrite(uint8(im),sprintf(opts.ims_crop,e,frame));
        
        coor = gt(m).stickmen.coor;
        P = size(coor,2);
        coor_new = coor + 1 - repmat([max(1,x1),max(1,y1)]',[2 P]);
        
        gt_modify(M).frame = frame;
        gt_modify(M).stickmen.coor = coor_new;
        
        testframe(M) = frame;
    end
    WriteStickmenAnnotationTxt(sprintf(opts.anno_crop,e),gt_modify);
    Numtest = M;
    save(sprintf('../data/buffy_s5e%d_crop_numtest.mat',e),'Numtest','testframe');
end