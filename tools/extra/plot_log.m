M0 = csvread('caffe.workstation.cv.log.INFO.20150425-104039.31275.test.csv',1,0);
M1 = csvread('caffe.workstation.cv.log.INFO.20150425-104121.32941.test.csv',1,0);
M = [M0;M1(2:end,:)];
h = figure;
subplot(1,3,1);plot(M0(:,1),M0(:,3));
xlim([0,600]);
legend('learning rate: 0.01','Location','southeast');legend('boxoff');
title('training part 1');
xlabel('training iterations');ylabel('val accuracy');
subplot(1,3,2);plot(M1(:,1),M1(:,3));
legend('learning rate: 0.001','Location','southeast');legend('boxoff');
title('training part 2');
xlabel('training iterations');ylabel('val accuracy');
subplot(1,3,3);plot(M(:,1),M(:,3));
legend('learning rate: the whole','Location','southeast');legend('boxoff');
%title('val accuracy v.s. training iterations');
title('the whole training process');
xlabel('training iterations');ylabel('val accuracy');
saveas(h,'cnn_accuracy_plot.eps','psc2')