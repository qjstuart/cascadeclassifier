I = imread('SC1.png');
[Gx,Gy] = imgradientxy(I);

figure
imshowpair(Gx,Gy,'montage')
title('Directional gradients Gx (horizontal) and Gy (vertical)')

[featureVector,hogVisualization] = extractHOGFeatures(I);
figure;
imshow(I)
hold on;
plot(hogVisualization)