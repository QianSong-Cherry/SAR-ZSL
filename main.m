
%main.m
%====================================
% EM Simulation-Aided Zero-Shot Learning for SAR 
% Automatic Target Recognition
% by Qian Song Mar. 1 2019
% ===================================
clear
close all

%% End-to-End SAR-ATR Cases
filename = '165';
% filename = '181';

% Target detection
BBox = target_detection(filename);

%==============================
% ==    Classify detected patches in Python   ==
%==============================

% Target classification
target_cla(filename, BBox);

