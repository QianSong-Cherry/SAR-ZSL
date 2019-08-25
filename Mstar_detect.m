function BBox = Mstar_detect(img)
%% Mstar detection
%% by Qian Guo, 2018/09/30
%% MSTAR 
thresh = mean(img(:));
[wid,len] = size(img);
img1 = img;
for i = 1:wid
    for j = 1:len
         if img(i,j) <= 3*thresh
            img1(i,j) = 0;
         end    
    end
end
% figure;imshow(img1);
%% 
pic_s = 88;
[L,~] = bwlabel(img1);
vehicle = bwpropfilt(logical(L), 'Area', [50,5000]); % figure;imshow(vehicle);
[L,~] = bwlabel(vehicle);
S = regionprops(L,'Centroid');
cen = cat(1,S.Centroid);
[len2,~] = size(cen);
x_min = [];x_max = [];
y_min = [];y_max = [];
for i = 1:len2
    x_min(i,1) = fix(cen(i,1) - pic_s/2);
    x_max(i,1) = fix(cen(i,1) + pic_s/2);
    y_min(i,1) = fix(cen(i,2) - pic_s/2);
    y_max(i,1) = fix(cen(i,2) + pic_s/2);
    if x_min(i,1) < 1
       x_min(i,1) = 1; 
       x_max(i,1) = pic_s; 
    end
    if y_min(i,1) < 1
       y_min(i,1) = 1; 
       y_max(i,1) = pic_s; 
    end
    if x_max(i,1) > len
       x_max(i,1) = len; 
       x_min(i,1) = len-pic_s+1; 
    end
    if y_max(i,1) > wid
       y_max(i,1) = wid; 
       y_min(i,1) = wid-pic_s+1;
    end
end
%% 
load('./data/mstar_net6.mat');
load('./data/Vehicle_label.mat');
bbox = [];
bbox(:,1) = x_min;
bbox(:,2) = y_min;
bbox(:,3) = x_max - x_min;
bbox(:,4) = y_max - y_min;
scores = [];
piclen = 20;
for i = 1:len2
    top_p = fix(cen(i,2)-piclen);
    if top_p < 1
        top_p = 1;
    end
    bottom_p = fix(cen(i,2)+piclen);
    if bottom_p > wid
        bottom_p = wid;
    end
    left_p = fix(cen(i,1)-piclen);
    if left_p < 1
        left_p = 1;
    end
    right_p = fix(cen(i,1)+piclen);
    if right_p > len
        right_p = len;
    end
    img_s = img(top_p:bottom_p,left_p:right_p);
    scores(i,1) = mean(img_s(:));
end

[bbox_new,~] = nms_cfar(bbox,scores,'Min',0.1);
[len3,~] = size(bbox_new);
count = 1;
BBox = [];
for i = 1:len3
    pic = img(bbox_new(i,2):bbox_new(i,2)+bbox_new(i,4),bbox_new(i,1):bbox_new(i,1)+bbox_new(i,3));
    pic = imresize(pic,[88,88]);
    [PLabel,~] = classify(net_Mstar_Two,pic);
    if PLabel == Vehicle
        BBox(count,:) = bbox_new(i,:);
        count = count+1;
    end
end

