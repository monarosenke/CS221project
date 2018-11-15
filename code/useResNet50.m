% for CS221 project, Dawn Finzi and Mona Rosenke

% step 1: using an existing, trained model (resnet 50)

clearvars
% load pretrained ResNet50
net = resnet50;
inputSize = net.Layers(1).InputSize;

% load images
files = dir('../data/**/*.jpg');
I = cell(size(files));
for i = 1:length(I)
    image = imread([files(i).folder '/' files(i).name]);
    I{i} = imresize(image,inputSize(1:2));
end


% loading meta data
D = readtable('../data/HAM10000_metadata.csv');
categories = unique(D.dx);
Y = zeros(size(D,1),1);
for i = 1:length(categories)
Y(find(strcmp(D.dx,categories{i}))) = i;
end




% classifying the image
[label,scores] = classify(net,I);
label
title(string(label) + ", " + num2str(100*scores(classNames == label),3) + "%");


%% display top predictions
[~,idx] = sort(scores,'descend');
idx = idx(5:-1:1);
classNamesTop = net.Layers(end).ClassNames(idx);
scoresTop = scores(idx);

figure
barh(scoresTop)
xlim([0 1])
title('Top 5 Predictions')
xlabel('Probability')
yticklabels(classNamesTop)


