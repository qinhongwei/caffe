function hdl = DrawStickman(Sticks, img)
% hdl = DrawStickmen(Sticks, img)
% Overlays segments in 'Sticks' on image 'img'
%
% Input:
%  - Sticks: matrix [4, nparts]. Sticks(:,i) --> (x1, x2, y1, y2)' 
%  - img: image to show. Can be [].
%
% Output:
%  - hdl: figure handler.
%
% See also ReadStickmenAnnotationTxt
%
% MJMJ/2008 modified by Eichner/2009
%

hdl = -1;

if nargin < 2,
   img = [];
end

if ~isempty(img),
   hdl = imshow(img);
   hold on
end

colors = sampleColors();

for p = 1:size(Sticks, 2),

   X = Sticks([1 3],p);
   Y = Sticks([2 4],p);   
   if any(X),
      hl = line(X', Y');
      set(hl, 'color', colors(p, :) );
      set(hl, 'LineWidth', 4);
      ht = text(5+mean(X), 5+mean(Y), num2str(p));
      set(ht, 'color', colors(p, :) );
      set(ht , 'FontWeight', 'bold');
   end

end


 
