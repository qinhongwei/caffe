function imagelist(dataPath)
% creat listfile of the input folder

fileList = dir([dataPath, '/*.jpg']);
imageNum = length(fileList);
dataName = [dataPath, '/list.txt'];
if(exist(dataName, 'file'))
   delete(dataName);
end;
fileID = fopen(dataName, 'w');
for i = 1:imageNum
   fprintf(fileID, '%s\n', fileList(i).name);
end;
fclose(fileID);
