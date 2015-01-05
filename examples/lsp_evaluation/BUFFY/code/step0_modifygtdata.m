close all; clear; clc;

startup;
 
load init.mat;

opts.anno_modify = '../data/buffy_s5e%d_sticks_modify.txt';

for e = opts.epi
    fprintf('episode %d/%d\n',e,max(opts.epi));
    gt = ReadStickmenAnnotationTxt(sprintf(opts.anno,e));
    load(sprintf(opts.labels,e));
    N = length(gt);
    clear gt_modify;
    for n = 1:N
        fprintf('%d/%d\n',n,N);
        head = gt(n).stickmen.coor(:,6);
        head = reshape(head,2,2);
        torso = gt(n).stickmen.coor(:,1);
        torso = reshape(torso,2,2);
        luarm = gt(n).stickmen.coor(:,2);
        luarm = reshape(luarm,2,2);
        ruarm = gt(n).stickmen.coor(:,3);
        ruarm = reshape(ruarm,2,2);
        llarm = gt(n).stickmen.coor(:,4);
        llarm = reshape(llarm,2,2);
        rlarm = gt(n).stickmen.coor(:,5);
        rlarm = reshape(rlarm,2,2);
        lsho = labels(13,:,n)';
        rsho = labels(15,:,n)';
        
        head_new = zeros(2,2);
        torso_new = zeros(2,2);
        luarm_new = zeros(2,2);
        ruarm_new = zeros(2,2);
        llarm_new = zeros(2,2);
        rlarm_new = zeros(2,2);
        
        dist = zeros(2);
        for i = 1:2
            for j = 1:2
                dist(i,j) = norm(head(:,i) - torso(:,j));
            end
        end
        [I J] = find(dist == min(dist(:)));
        
        head_new(:,1) = head(:,3-I);
        head_new(:,2) = head(:,I);
        torso_new(:,1) = torso(:,J);
        torso_new(:,2) = torso(:,3-J);
        % ------------    
%         dist = zeros(2);
%         for i = 1:2
%             for j = 1:2
%                 dist(i,j) = norm(luarm(:,i) - llarm(:,j));
%             end
%         end
%         [I J] = find(dist == min(dist(:)));
%         
%         luarm_new(:,1) = luarm(:,3-I);
%         luarm_new(:,2) = luarm(:,I);
%         llarm_new(:,1) = llarm(:,J);
%         llarm_new(:,2) = llarm(:,3-J);        
%         
%         dist = zeros(2);
%         for i = 1:2
%             for j = 1:2
%                 dist(i,j) = norm(ruarm(:,i) - rlarm(:,j));
%             end
%         end
%         [I J] = find(dist == min(dist(:)));
%         
%         ruarm_new(:,1) = ruarm(:,3-I);
%         ruarm_new(:,2) = ruarm(:,I);
%         rlarm_new(:,1) = rlarm(:,J);
%         rlarm_new(:,2) = rlarm(:,3-J);
        % -------------
        dist = zeros(1,2);

        dist(1) = norm(luarm(:,1) - lsho);
        dist(2) = norm(luarm(:,2) - lsho);
        
        [dummy I] = min(dist);
        luarm_new(:,1) = luarm(:,I);
        luarm_new(:,2) = luarm(:,3-I);

        dist(1) = norm(ruarm(:,1) - rsho);
        dist(2) = norm(ruarm(:,2) - rsho);
        
        [dummy I] = min(dist);
        ruarm_new(:,1) = ruarm(:,I);
        ruarm_new(:,2) = ruarm(:,3-I);
        
        dist(1) = norm(llarm(:,1) - luarm(:,2));
        dist(2) = norm(llarm(:,2) - luarm(:,2));
        
        [dummy I] = min(dist);
        llarm_new(:,1) = llarm(:,I);
        llarm_new(:,2) = llarm(:,3-I);        
        
        dist(1) = norm(rlarm(:,1) - ruarm(:,2));
        dist(2) = norm(rlarm(:,2) - ruarm(:,2));
        
        [dummy I] = min(dist);
        rlarm_new(:,1) = rlarm(:,I);
        rlarm_new(:,2) = rlarm(:,3-I);        
        % -------------

        head_new = reshape(head_new,4,1);
        torso_new = reshape(torso_new,4,1);
        luarm_new = reshape(luarm_new,4,1);
        ruarm_new = reshape(ruarm_new,4,1);
        llarm_new = reshape(llarm_new,4,1);
        rlarm_new = reshape(rlarm_new,4,1);
        
        gt_modify(n).frame = gt(n).frame;
        gt_modify(n).stickmen.coor = [torso_new luarm_new ruarm_new llarm_new rlarm_new head_new];
    end
    WriteStickmenAnnotationTxt(sprintf(opts.anno_modify,e),gt_modify);
end