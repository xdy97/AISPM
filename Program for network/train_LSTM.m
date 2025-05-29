
clear
netCNN = googlenet;

dataFolder = "LabelVideo";
[files, labels] = hmdb51Files(dataFolder);

inputSize = netCNN.Layers(1).InputSize(1:2);
layerName = "pool5-7x7_s1";

tempFile = fullfile(tempdir, "video0919.mat");

if exist(tempFile, 'file')
    load(tempFile, "sequences");
else
    numFiles = numel(files);
    sequences = cell(numFiles, 1);

    for i = 1:numFiles
        fprintf("Reading file %d of %d...\n", i, numFiles);

        video = readVideo(files(135));

        video = centerCrop(video, inputSize);

        sequences{i} = activations(netCNN, video, layerName, 'OutputAs', 'columns');
    end

    save(tempFile, "sequences", "-v7.3");
end

numObservations = numel(sequences);
idx = randperm(numObservations);
N = floor(0.9 * numObservations);

idxTrain = idx(1:N);
sequencesTrain = sequences(idxTrain);
labelsTrain = labels(idxTrain);

idxValidation = idx(N+1:end);
sequencesValidation = sequences(idxValidation);
labelsValidation = labels(idxValidation);

inputSizeLSTM = size(sequencesTrain{1}, 1);  
numClasses = numel(categories(labelsTrain));  

layers = [ ...
    sequenceInputLayer(inputSizeLSTM)  
    lstmLayer(100, 'OutputMode', 'last')  
    fullyConnectedLayer(numClasses)  
    softmaxLayer  
    classificationLayer];  

options = trainingOptions('adam', ...
    'MaxEpochs', 20, ...  
    'MiniBatchSize', 32, ...
    'ValidationData', {sequencesValidation, labelsValidation}, ...  
    'ValidationFrequency', 10, ...  
    'Verbose', true, ...
    'Shuffle', 'every-epoch', ...
    'ExecutionEnvironment', 'auto', ...  
    'Plots', 'training-progress');  

[netLSTM, trainInfo] = trainNetwork(sequencesTrain, labelsTrain, layers, options);

trainingLoss = trainInfo.TrainingLoss;
validationLoss = trainInfo.ValidationLoss;
trainingAccuracy = trainInfo.TrainingAccuracy;
validationAccuracy = trainInfo.ValidationAccuracy;

iterations = 1:numel(trainingLoss);

figure;
subplot(4, 1, 1);
plot(iterations, trainingLoss, 'b', 'LineWidth', 1.5);
hold on;
plot(iterations, validationLoss, 'r--', 'LineWidth', 1.5);
xlabel('Iteration');
ylabel('Loss');
legend('Training Loss', 'Validation Loss');
title('Training and Validation Loss');
grid on;

subplot(4, 1, 2);
plot(iterations, trainingAccuracy, 'b', 'LineWidth', 1.5);
hold on;
plot(iterations, validationAccuracy, 'r--', 'LineWidth', 1.5);
xlabel('Iteration');
ylabel('Accuracy (%)');
legend('Training Accuracy', 'Validation Accuracy');
title('Training and Validation Accuracy');
grid on;

        video = readVideo(files(135));

        video = centerCrop(video, inputSize);
 
        sequences{end} = activations(netCNN, video, layerName, 'OutputAs', 'columns');
sequencesValidation = [sequencesValidation;sequences{end}];
labelsValidation(end+1) = "Block";

[predictedLabels, scores] = classify(netLSTM, sequencesValidation);  

figure
cm = confusionchart(labelsValidation, predictedLabels);
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';
