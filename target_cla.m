
function target_cla(filename, BBox)

load('./result/pred.mat')      
[~,pred] = max(label_pred,[],2);
if strcmp(filename,'181')    
    load('./data/HB06181_with_mstar.mat');
else
    load('./data/HB06165_with_mstar.mat');
end
img = imgdata;
%% results
label_name = {'BRDM2','BTR60','D7','2S1','T62','ZIL131','ZSU234','T72','BMP2','BTR70'};
if strcmp(filename,'181')
    positions = [89, 63, 128, 128;
                 275, 124, 128, 128;
                 497, 43, 128, 128;
                 683, 109, 128, 128;
                 941, 39, 128, 128;
                 1183, 49, 128, 128;
                 59, 457, 128, 128;
                 120, 600, 128, 128;
                 39, 740, 128, 128;
                 289, 761, 128, 128;
                 461, 773, 128, 128;
                 553, 421, 128, 128;
                 717, 379, 128, 128;
                 940, 331, 128, 128;
                 653, 531, 128, 128;
                 895, 467, 128, 128;
                 468, 595, 128, 128;
                 917, 571, 128, 128;
                 768, 703, 128, 128;
                 957, 663, 128, 128;];
else
    positions = [195, 423, 128, 128;
                 443, 427, 128, 128;
                 295, 633, 128, 128;
                 465, 669, 128, 128;
                 848, 723, 128, 128;
                 163, 847, 128, 128;
                 397, 893, 128, 128;
                 742, 959, 128, 128;
                 1115, 1045, 128, 128;
                 151, 1085, 128, 128;
                 1298, 1069, 128, 128;
                 473, 1097, 128, 128;
                 801, 1195, 128, 128;
                 1312, 1227, 128, 128;
                 251, 1345, 128, 128;
                 767, 1449, 128, 128;
                 1030, 1501, 128, 128;
                 453, 1597, 128, 128;
                 801, 1654, 128, 128;
                 1008, 1660, 128, 128;] - [64, 64, 0, 0];
end
fg = figure();
pos = get(fg, 'position');
set(fg, 'position', [pos(1) pos(2) size(img,2)/5 size(img,1)/5]);
set(gca, 'units', 'pixel');
set(gca, 'position', [0 0 size(img,2)/5 size(img,1)/5]);
imshow(imadjust(img,[0,0.3])); hold on

is_correct = zeros(size(BBox,1), 1);
for i = 1:size(BBox,1)
    [~,idx] = min(abs(positions(:,1) - BBox(i,1))+abs(positions(:,2) - BBox(i,2)));
    is_correct(i) = (y_part(idx)==pred(i));
end
disp(['Accuracy: ', num2str(sum(is_correct)./20)])

for i = 1:size(positions,1)
    text(positions(i,1)+30,positions(i,2)+100,label_name{y_part(i)},'Color','cyan','FontSize',5);
end

for i = 1:size(BBox,1)
    if is_correct(i)==1
        text(BBox(i,1)+30,BBox(i,2)+30,label_name{pred(i)},'Color','green','FontSize',5);
        rectangle('Position',BBox(i,:),'EdgeColor','green')
    else
        text(BBox(i,1)+30,BBox(i,2)+30,label_name{pred(i)},'Color','red','FontSize',5);
        rectangle('Position',BBox(i,:),'EdgeColor','red')
    end
end
