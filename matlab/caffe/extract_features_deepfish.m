%% vgg / caffe spec

use_gpu = 1;
caffe('set_device', 0);
model_def_file = '/home/cv/image-net/caffe/models/deepfish_jin_right/deploy.prototxt';
model_file = '/home/cv/image-net/caffe/models/deepfish_jin_right/caffenet_train_full_iter_65000.caffemodel';
batch_size = 10;

matcaffe_init(use_gpu, model_def_file, model_file);

%% input files spec

save_path = '/media/cv/Data/fishData/fish_boundingbox_47_right/extractfeatures/';
data_path = '/media/cv/Data/fishData/fish_boundingbox_47_right/test/';
fs = textread(['/media/cv/Data/fishData/fish_boundingbox_47_right/', 'testlist.txt'], '%s');
N = length(fs);

%%
load('deepfish_mean.mat')
% iterate over the iamges in batches
feats = zeros(43, N, 'single');
for b=1:batch_size:N

    % enter images, and dont go out of bounds
    Is = {};
    for i = b:min(N,b+batch_size-1)
        I = imread([data_path, fs{i}]);
        if ndims(I) == 2
            I = cat(3, I, I, I); % handle grayscale edge case. Annoying!
        end
        Is{end+1} = I;
    end
    %input_data = prepare_images_batch_extract_features(Is);
    input_data = prepare_batch_deepfish(Is,deepfish_mean,batch_size);

    tic;
    scores = caffe('forward', {input_data});
    scores = squeeze(scores{1});
    tt = toc;

    nb = length(Is);
    feats(:, b:b+nb-1) = scores(:,1:nb);
    fprintf('%d/%d = %.2f%% done in %.2fs\n', b, N, 100*(b-1)/N, tt);
end

%% write to file

save([save_path 'test134_vgg_scores_hdf5.mat'], 'feats', '-v7.3');
save([save_path 'test134_vgg_scores.mat'], 'feats');
