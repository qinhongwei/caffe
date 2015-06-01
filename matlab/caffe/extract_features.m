%% vgg / caffe spec

use_gpu = 1;
caffe('set_device', 0);
model_def_file = '/home/cv/image-net/caffe/models/VGG_ILSVRC_16_layers/feature_extraction_deploy.prototxt';
model_file = '/home/cv/image-net/caffe/models/VGG_ILSVRC_16_layers/VGG_ILSVRC_16_layers.caffemodel';
batch_size = 10;

matcaffe_init(use_gpu, model_def_file, model_file);

%% input files spec

save_path = '/home/cv/image-net/caffe/examples/depth_estimation/';
data_path = '/media/cv/Data/depth_estimation/Test134Img/';
fs = textread([data_path 'list.txt'], '%s');
N = length(fs);

%%

% iterate over the iamges in batches
feats = zeros(4096, N, 'single');
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
    input_data = prepare_images_batch_extract_features(Is);

    tic;
    scores = caffe('forward', {input_data});
    scores = squeeze(scores{1});
    tt = toc;

    nb = length(Is);
    feats(:, b:b+nb-1) = scores(:,1:nb);
    fprintf('%d/%d = %.2f%% done in %.2fs\n', b, N, 100*(b-1)/N, tt);
end

%% write to file

save([save_path 'test134_vgg_feats_hdf5.mat'], 'feats', '-v7.3');
save([save_path 'test134_vgg_feats.mat'], 'feats');
