function test = boxTest(box,points)
% boxTest(box,points)
% test weather point lies in the box
% Input:
%  point(:,n) = [x y]'
%  box = [minx miny maxx maxy]
% Output:
%   test = bool(1,n) - true/false value for each point passed to the routine
%
points = points';
test = (points(:,1) > box(1) & points(:,1) < box(3)) | (points(:,2) > box(2) & points(:,2) < box(4));
test = test';