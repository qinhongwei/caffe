clear;clc;
list_im = '/media/cv/Data/trafficSignRGB/test3.txt';
use_gpu = 1;

for i = 1:5
    model_n = i;
    if(i==1)
        deploy_name = '../../models/trafficSignRGB_jin/deploy.prototxt';
        model_name = '../../models/trafficSignRGB_jin/caffenet_train_full_iter_21000.caffemodel';
    else
        deploy_name = ['../../models/trafficSignRGB_jin', num2str(i), '/deploy.prototxt'];
        model_name = ['../../models/trafficSignRGB_jin', num2str(i), '/caffenet_train_full_iter_21000.caffemodel'];
    end; %end if
    [scores, list_im_cell, real_label, accuracy] = matcaffe_batch(list_im,use_gpu,model_n, deploy_name, model_name);
    scores_all(:,:,i) = scores;
    accuracy_all(i) = accuracy;
end; % end for
scores_ensemble = mean(scores_all, 3);
[~, maxlabel] = max(scores_ensemble);
predict_label = int32(maxlabel-1);
right_label = (real_label == predict_label);
accuracy_ensemble = sum(right_label)/length(right_label);
fprintf('\n\t');
accuracy_all
fprintf('The accuracy after ensemble is %f\n', accuracy_ensemble);

% save the vars
save([list_im, '.accuracy.mat'], 'scores_all', 'accuracy_all','predict_label', 'real_label', 'list_im_cell','accuracy_ensemble', 'scores_ensemble');
