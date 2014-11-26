close all; clear; clc;

load init.mat;

colors = sampleColors();

for e = opts.epi
    fprintf('episode %d/%d\n',e,max(opts.epi));
    gt = ReadStickmenAnnotationTxt(sprintf(opts.anno,e));
    load(sprintf(opts.labels_parse,e));
    
    N = length(gt);
    for n = 1:N
        fprintf('%d/%d\n',n,N);
        img = imread(sprintf(opts.ims,e,gt(n).frame));
        figure(1); clf; axis image;
        pts = labels(:,:,n);
        
        Sticks(1,:) = [pts(1,:) pts(2,:)];
        Sticks(2,:) = [pts(2,:) pts(3,:)];
        Sticks(3,:) = [pts(2,:) pts(5,:)];
        Sticks(4,:) = [pts(3,:) pts(4,:)];
        Sticks(5,:) = [pts(5,:) pts(6,:)];
        Sticks(6,:) = [pts(4,:) pts(7,:)];
        Sticks(7,:) = [pts(6,:) pts(8,:)];
        Sticks(8,:) = [pts(3,:) pts(9,:)];
        Sticks(9,:) = [pts(5,:) pts(10,:)];
        
        imshow(img); hold on;
        for p = 1:size(Sticks, 1)

            X = Sticks(p,[1 3]);
            Y = Sticks(p,[2 4]);   
       
            hl = line(X, Y);
            set(hl, 'color', colors(p, :) );
            set(hl, 'LineWidth', 4);
            ht = text(5+mean(X), 5+mean(Y), num2str(p));
            set(ht, 'color', colors(p, :) );
            set(ht , 'FontWeight', 'bold');
        end  
        pause;
    end
    pause;
end
