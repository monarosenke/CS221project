% CS221 project, Dawn Finzi and Mona Rosenke
%
% Using an existing, trained model (resnet 50) and replacing the
% last layer with a new layer and our images.
%
% This code is started using the documentation for Matlab 2018b by
% MathWorks on how to start using neural networks. From there we modified
% the code to use it for our project using the skin cancer dataset from
% kaggel.

clearvars
% load pretrained ResNet50
net = resnet50;
analyzeNetwork(net)

inputSize = net.Layers(1).InputSize;

% load our dataset
imds = imageDatastore('../data/','FileExtensions','.jpg','IncludeSubfolders',true);

% files = dir('../data/**/*.jpg');
% I = cell(size(files));
% for i = 1:length(I)
%     image = imread([files(i).folder '/' files(i).name]);
%     I{i} = imresize(image,inputSize(1:2));
% end


% loading meta data
D = readtable('../data/HAM10000_metadata.csv');
numClasses = numel(unique(D.dx));
Y = zeros(size(D,1),1);
for i = 1:length(categories)
    Y(find(strcmp(D.dx,categories{i}))) = i;
end
% imds.Labels = Y;
imds.Labels = categorical(D.dx);

% splitting data
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7);


% replacing last fully connected layer in the network as well as the
% classification layer
lgraph = layerGraph(net);

newLearnableLayer = fullyConnectedLayer(numClasses, ...
    'Name','new_fc', ...
    'WeightLearnRateFactor',10, ...
    'BiasLearnRateFactor',10);

lgraph = replaceLayer(lgraph,'fc1000',newLearnableLayer);

newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,'ClassificationLayer_fc1000',newClassLayer);

% check that new layers are correctly connected:
figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
plot(lgraph)
ylim([0,10])


% freeze initial 10 layers
layers = lgraph.Layers;
connections = lgraph.Connections;

for i = 1:10
    if isprop(layers(i),'WeightLearnRateFactor')
        layers(i).WeightLearnRateFactor = 0;
    end
    if isprop(layers(i),'WeightL2Factor')
        layers(i).WeightL2Factor = 0;
    end
    if isprop(layers(i),'BiasLearnRateFactor')
        layers(i).BiasLearnRateFactor = 0;
    end
    if isprop(layers(i),'BiasL2Factor')
        layers(i).BiasL2Factor = 0;
    end
end

% create new layer graph
lgraph = layerGraph();
for i = 1:numel(layers)
    lgraph = addLayers(lgraph,layers(i));
end

for c = 1:size(connections,1)
    lgraph = connectLayers(lgraph,connections.Source{c},connections.Destination{c});
end

% data augmentation to prevent overfitting using our new images
pixelRange = [-30 30];
scaleRange = [0.9 1.1];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange, ...
    'RandXScale',scaleRange, ...
    'RandYScale',scaleRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);

augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

% making sure we learn fast in the now layers but slow in the early
% (existing) ones
options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',3e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'Plots','training-progress');

% TRAIN THE NETWORK
net = trainNetwork(augimdsTrain,lgraph,options);




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


