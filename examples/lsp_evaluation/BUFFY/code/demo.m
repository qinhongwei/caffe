close all; clear; clc;
img = imread('000063.jpg');
lF = ReadStickmenAnnotationTxt('../data/buffy_s5e2_sticks.txt');
hdl = DrawStickman(lF(1).stickmen.coor, img);
gt2 = ReadStickmenAnnotationTxt('../data/buffy_s5e2_sticks.txt','episode','2');
gt5 = ReadStickmenAnnotationTxt('../data/buffy_s5e5_sticks.txt','episode','5');
gt6 = ReadStickmenAnnotationTxt('../data/buffy_s5e6_sticks.txt','episode','6');
GTALL = [gt2(:); gt5(:); gt6(:)]';
load('../BMVC09best256.mat');
[detRate PCP] = BatchEvalBuffy(BMVC09best256,GTALL)