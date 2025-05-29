clc;clear;close all
load data.mat

numChannels = size(data{1},2)

numObservations = numel(data);
idxTrain = 1:floor(0.9*numObservations);
idxTest = floor(0.9*numObservations)+1:numObservations;
dataTrain = data(idxTrain);
dataTest = data(idxTest);

numObservationsTrain = numel(dataTrain);
XTrain = cell(numObservationsTrain,1);
TTrain = cell(numObservationsTrain,1);
for n = 1:numObservationsTrain
    X = dataTrain{n};
    XTrain{n} = X(1:end-1,:)*0.9;
    TTrain{n} = X(2:end,:)*0.9;
end

muX = mean(cell2mat(XTrain));
sigmaX = std(cell2mat(XTrain),0);

muT = mean(cell2mat(TTrain));
sigmaT = std(cell2mat(TTrain),0);

for n = 1:numel(XTrain)
    XTrain{n} = (XTrain{n} - muX) ./ sigmaX;
    TTrain{n} = (TTrain{n} - muT) ./ sigmaT;
end

% Define BiLSTM Neural Network Architecture
layers = [
    sequenceInputLayer(numChannels)
    bilstmLayer(128)          
    % batchNormalizationLayer  
    bilstmLayer(128)          
    % batchNormalizationLayer  
    dropoutLayer(0.2)       
    bilstmLayer(64)           
    % batchNormalizationLayer  
    fullyConnectedLayer(numChannels)];

% Specify Training Options
options = trainingOptions("adam", ...
    MaxEpochs=22, ...
    SequencePaddingDirection="left", ...
    Shuffle="every-epoch", ...
    Plots="training-progress", ...
    Verbose=false);

% Train Recurrent Neural Network
% net = trainnet(XTrain,TTrain,layers,"mse",options);
net = trainnet(XTrain,TTrain,layers,"mae",options);
% [netLSTM,info] = trainNetwork(XTrain,TTrain,layers,options);
numObservationsTest = numel(dataTest);
XTest = cell(numObservationsTest,1);
TTest = cell(numObservationsTest,1);
for n = 1:numObservationsTest
    X = dataTest{n};
    XTest{n} = (X(1:end-1,:) - muX) ./ sigmaX;
    TTest{n} = (X(2:end,:) - muT) ./ sigmaT;
end

YTest = minibatchpredict(net,XTest, ...
    SequencePaddingDirection="left", ...
    UniformOutput=false);



for idx = 1:6
Y = XTest{idx};
T = YTest{idx};

%% 评估误差
for n = 1:numObservationsTest
    % T = TTest{n};
    Ypred = minibatchpredict(net, XTest(n), ...
        SequencePaddingDirection = "left", ...
        UniformOutput = false);

    sequenceLength = size(T,1);
    % Y = Ypred{1}(end-sequenceLength+1:end,:);
    err(n) = rmse(Y, T, "all");
end
RMSE = mean(err);
% disp("Test RMSE = " + mean(err));
MAE = mean(abs(Y - T));
MSE = mean((Y - T).^2);
MAPE = mean(abs((Y - T) ./ T)) * 100;
SMAPE = mean(2 * abs(T - Y) ./ (abs(T) + abs(Y))) * 100;

SSE = sum((T - Y).^2);                     
SST = sum((T - mean(T)).^2);               
R2 = 1 - SSE / SST;

true_diff = diff(T);
pred_diff = diff(Y);

correct_direction = (sign(true_diff) == sign(pred_diff));

DA = sum(correct_direction) / length(correct_direction) * 100;

ResultTable = table(MSE,RMSE,MAE,MAPE,SMAPE,R2,DA);

% Signal_Store(idx*2-1,1:3984) = normalize(Signal_X);
% Signal_Store(idx*2,401:3984) = normalize(Signal_Y);
Signal_Store(idx*2-1,1:3983) = T;
Signal_Store(idx*2,1:3983) = Y;
DA_Store(idx) = DA;
R2_Store(idx) = R2;
SMAPE_Store(idx) = SMAPE;
MAPE_Store(idx) = MAPE;
MSE_Store(idx) = MSE;
MAE_Store(idx) = MAE;
RMSE_Store(idx) = RMSE;
end
PINGJIA = table(RMSE_Store',MAE_Store',MSE_Store',MAPE_Store',SMAPE_Store',R2_Store',DA_Store');

save traindata