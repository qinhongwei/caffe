function [name, label] = read_cell_my(fileName)
% fileName = '/media/cv/Data/trafficSignRGB/test.txt';
fileID = fopen(fileName);
C = textscan(fileID, '%s %d');
fclose(fileID);
name = C{1};
label = C{2};