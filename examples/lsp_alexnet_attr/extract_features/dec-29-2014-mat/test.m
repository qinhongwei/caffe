load('test_labels_joint_x_1000.mat');
x = feats;

load test_labels_joint_y_1000.mat
y = feats;


label_test = [x y]';
label_test = label_test(:, 1:3000);

filepath = '/home/wyang/Code/PE1.41DBN_human_detector/LSP/cache_test_top5/hdf5-mean/0001.h5';
h5disp(filepath);
info = h5info(filepath)
data = h5read(filepath,'/data');
label = h5read(filepath,'/label');

diff = label - label_test;
[h, w, c, n] = size(data)
for i = 1:n
    im      = data(:, :, :, i);
    points  = [label(1:14, i) label(15:end, i)]'*227;
    visualize_pose(im, points, ones(1, 14));
end