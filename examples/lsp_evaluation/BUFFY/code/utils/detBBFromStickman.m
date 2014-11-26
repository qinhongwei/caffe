function bb = detBBFromStickman(stick,classname,params)
% function minBB(stick,classname)
% stick in format rows: [x1; y1; x2; y2] cols: nsticks
% returns bb in format [minx miny maxx maxy]
% so far just for ubf
% requires params.det_hwratio(classid)
stick = double(stick);
classid = class_name2id(classname);
switch classname
  case 'ubf'
    %minx = min([stick(1,[ 1 2 3 6]) stick(3,[1 2 3 6])]); % min x of all parts besides lower arms
    %maxx = max([stick(1,[2 3]) stick(3,[2 3])]); % max x of all parts besides lower arms
    torso_center = [(stick(1,1)+stick(3,1))/2 (stick(2,1)+stick(4,1))/2]; %center [x y]
    miny = min([stick(2,[1 6]) stick(4,[1 6]) torso_center(2)]); % min y of torso and head
    maxy = max([stick(2,6) stick(4,6) torso_center(2) ]); %max y of head and torso center
    diffx = abs(miny-maxy)/params.det_hwratio(classid);
    head_center = [(stick(1,6)+stick(3,6))/2 (stick(2,6)+stick(4,6))/2];
    minx = (head_center(1)+torso_center(1))/2 - diffx/2;
    maxx = (head_center(1)+torso_center(1))/2 + diffx/2;
  case 'ubp'
    error('not written yet');
  case 'full'
    error('not written yet');
end
bb = [minx miny maxx maxy];
end