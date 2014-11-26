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
    
    figure(1); clf;  axis image; imshow(img);
    if evalNo 
%         DrawStickman(stick(evalNo).coor, img);
        DrawStickman(gt(m).stickmen.coor,img);

        pars.det_hwratio(2) = 0.9;
        pars.iou_thresh = 0.5;
        gt_bb = detBBFromStickman(gt(m).stickmen.coor,'ubf',pars);
%         box = stick(evalNo).det;
        box = gt_bb;
        x1 = box(1);
        y1 = box(2);
        x2 = box(3);
        y2 = box(4);
        line([x1 x1 x2 x2 x1]',[y1 y2 y2 y1 y1]','color','g','linewidth',4);
        
%         box = stick(evalNo).det;
        box = gt_bb;
        x1 = box(1);
        y1 = box(2);
        x2 = box(3);
        y2 = box(4);
        Width = x2 - x1; Height = y2 - y1;
        x1 = x1 - Width*7/10;
        y1 = y1 - Height*2/3;
        x2 = x2 + Width*7/10;
        y2 = y2 + Height*12/10;
        line([x1 x1 x2 x2 x1]',[y1 y2 y2 y1 y1]','color','r','linewidth',4);
        
        
    end
end