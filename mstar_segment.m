function [shadow,target] = mstar_segment(img)
bg_mask = true(128, 128);
bg_mask(64 - 30 : 64 + 30, 64 - 30 : 64 + 30) = 0;
bg_data = img(bg_mask);
bg_data = bg_data(bg_data > 0);
pfa_upper = 5e-3; %0.5e-1; % 
pfa_lower = 1.5e-1; %
ratio = 0.1;
phat = gamfit(double(bg_data));
a = phat(1);
b = phat(2);
th_lower = gaminv(pfa_lower, a, b);
th_upper = gaminv(1 - pfa_upper, a, b);
fg = img > th_upper;
sh = img < th_lower;
fg = imclose(fg, strel('disk', 1));
fg = bwpropfilt(fg,'Area',1);
sh = bwpropfilt(sh, 'Area', 1);
B = sort(img(:), 'ascend');
if sum(fg(:)) > ratio * 128 * 128
    th_upper = B(128 * 128 - round(ratio * 128 * 128));
    fg = img > th_upper;
end
if sum(sh(:)) > ratio * 128 * 128
    th_lower = B(round(ratio * 128 * 128));
    sh = img < th_lower;
end
fg = bwpropfilt(fg,'Area',1);
% fg = imdilate(fg,strel('disk',2));
fg = imfill(fg, 'holes');
stat = regionprops(fg, 'ConvexImage', 'BoundingBox');
if find(fg~=0)
    bbox = ceil(stat.BoundingBox);
    cvex = stat.ConvexImage;
    fg(bbox(2) : bbox(4) + bbox(2) - 1, bbox(1) : bbox(1) + bbox(3) - 1) = cvex;
    sh = bwpropfilt(sh, 'Area', 1);
    % sh = imdilate(sh,strel('disk',2));
    sh = imfill(sh, 'holes');
    stat = regionprops(sh, 'ConvexImage', 'BoundingBox');
    bbox = ceil(stat.BoundingBox);
    cvex = stat.ConvexImage;
    sh(bbox(2) : bbox(4) + bbox(2) - 1, bbox(1) : bbox(1) + bbox(3) - 1) = cvex;
    target = fg;
    shadow = sh & ~fg;
else
    shadow = false(128, 128);
    target = true(128, 128);
end
end
