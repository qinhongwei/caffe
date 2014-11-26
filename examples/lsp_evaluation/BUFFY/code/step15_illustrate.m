close all; clear; clc;

startup;

load init.mat;
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
    figure(1); clf;  axis image; 
    stick = BMVC09best256(n).stickmen;
    for i = 1:length(stick)
        DrawStickman(stick(i).coor, img);
        box = stick(i).det;
        x1 = box(1);
        y1 = box(2);
        x2 = box(3);
        y2 = box(4);
        line([x1 x1 x2 x2 x1]',[y1 y2 y2 y1 y1]','color','g','linewidth',4);
    end
    
    for i = 1:length(stick)
        box = stick(i).det;
        x1 = box(1);
        y1 = box(2);
        x2 = box(3);
        y2 = box(4);
        Width = x2 - x1; Height = y2 - y1;
        x1 = x1 - Width/5;
        y1 = y1 - Height/10;
        x2 = x2 + Width/5;
        y2 = y2 + Height/2 + Height/10;
        line([x1 x1 x2 x2 x1]',[y1 y2 y2 y1 y1]','color','r','linewidth',4);
    end
        
    pause;
end