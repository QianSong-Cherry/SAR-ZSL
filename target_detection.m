
function BBox = target_detection(filename)
%% read data -> img label_pred
if strcmp(filename,'181')
    load('./data/HB06181_with_mstar.mat');
else
    load('./data/HB06165_with_mstar.mat');
end
img = imgdata;
fg = figure();
pos = get(fg, 'position');
set(fg, 'position', [pos(1) pos(2) size(img,2)/5 size(img,1)/5]);
set(gca, 'units', 'pixel');
set(gca, 'position', [0 0 size(img,2)/5 size(img,1)/5]);
imshow(img,[0 0.3]);

%% target detection
BBox = Mstar_detect(uint8(imadjust(img,[0 0.3])*270));
BBox(:,1:2) = BBox(:,1:2) - 20;
BBox(:,3:4) = 128;
fg = figure();
pos = get(fg, 'position');
set(fg, 'position', [pos(1) pos(2) size(img,2)/5 size(img,1)/5]);
set(gca, 'units', 'pixel');
set(gca, 'position', [0 0 size(img,2)/5 size(img,1)/5]);
img1 = insertShape(img,'rectangle',BBox,'LineWidth',5);
imshow(imadjust(img1,[0,0.3])); hold on
detected_targets = zeros(size(BBox,1),128,128);
detected_targets_seg = zeros(size(BBox,1),128,128);
for i = 1:size(BBox,1)
    detected_targets(i,:,:) = img(BBox(i,2)+(1:128),BBox(i,1)+(1:128));   
    [shadow,target] = mstar_segment(squeeze(detected_targets(i,:,:)));
    detected_targets_seg(i,:,:) = double(shadow | target); 
end
L = 1;
for i = 1:size(BBox,1)
    temp = squeeze(detected_targets(i,:,:));
    detected_targets(i,:,:) = imadjust(FANS(temp,L),[0 3*mean(temp(:))]);   
end
data_test = detected_targets.*detected_targets_seg;

if strcmp(filename,'181')
    save('./data/test_181.mat','data_test');
else
    save('./data/test_165.mat','data_test');
end
