% CS221 project, Dawn Finzi and Mona Rosenke
%
% Using an existing, trained model (resnet 50) and replacing the
% last layer with a new layer and our images.
%
% This code is started using the documentation for Matlab 2018b by
% MathWorks on how to start using neural networks. From there we modified
% the code to use it for our project using the skin cancer dataset from
% kaggel.
%
% Code assumes that pwd is ~/CS211project/code 

clearvars

% if miniset, only the first 100 images of the skin cancer dataset will be used to
% retrain the last layer
miniset = 1
imageFiltering = 'g'; % e for edge, g for gradient, n for none


%% load pretrained ResNet50
net = resnet50;
analyzeNetwork(net);

inputSize = net.Layers(1).InputSize;



%% load our dataset

switch imageFiltering
    case 'n'
        imds = imageDatastore('../data/','FileExtensions','.jpg','IncludeSubfolders',true);
    case 'e'
        imds = imageDatastore('../processedImages/edgeFiltered/','FileExtensions','.jpg','IncludeSubfolders',true);
    case 'g'
        imds = imageDatastore('../processedImages/gradientFiltered/','FileExtensions','.jpg','IncludeSubfolders',true);
end

% filtering 
%switch filtering
    %case 'e'
        imds




% loading meta data
D = readtable('../data/HAM10000_metadata.csv');
numClasses = numel(unique(D.dx));
% Y = zeros(size(D,1),1);
% for i = 1:length(categories)
%     Y(find(strcmp(D.dx,categories{i}))) = i;
% end
% imds.Labels = Y;
imds.Labels = categorical(D.dx);

% splitting data into train and validation set, if only a mini test dataset, use
% a small subset of the full dataset
if miniset
    [imdsTrain,imdsValidation,~] = splitEachLabel(imds,0.01,0.004);
else
    [imdsTrain,imdsValidation] = splitEachLabel(imds,0.7);
end

%% replacing last fully connected layer in the network as well as the classification layer
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

% making sure we learn fast in the new layers but slow in the early
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

%% TRAIN THE NETWORK
net = trainNetwork(augimdsTrain,lgraph,options);


%% classifying the image

[YPred,probs] = classify(net,augimdsValidation);
accuracy = mean(YPred == imdsValidation.Labels)

idx = randperm(numel(imdsValidation.Files),4);
figure
for i = 1:4
    subplot(2,2,i)
    I = readimage(imdsValidation,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label) + ", " + num2str(100*max(probs(idx(i),:)),3) + "%");
end
