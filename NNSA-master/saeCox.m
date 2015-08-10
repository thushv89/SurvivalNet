clear all
prepareData

K = 100;

%% configure secondary network
% Create an empty network
autoencHid1 = network;

% Set the number of inputs and layers
autoencHid1.numInputs = 1;
autoencHid1.numlayers = 1;

% Connect the 1st (and only) layer to the 1st input, and also connect the
% 1st layer to the output
autoencHid1.inputConnect(1,1) = 1;
autoencHid1.outputConnect = 1;

% Add a connection for a bias term to the first layer
autoencHid1.biasConnect = 1;

%% configure autoencoder
%hiddenSize1 = 100;
saeC = zeros(80,1);
for hiddenSize = 80:2:80
autoenc1 = feedforwardnet(hiddenSize);
autoenc1.trainFcn = 'trainscg';
autoenc1.trainParam.epochs = 200;

% Do not use process functions at the input or output
autoenc1.inputs{1}.processFcns = {};
autoenc1.outputs{2}.processFcns = {};

% Set the transfer function for both layers to the logistic sigmoid
autoenc1.layers{1}.transferFcn = 'logsig';
autoenc1.layers{2}.transferFcn = 'logsig';

% Divide samples into three sets randomly
autoenc1.divideFcn = 'dividetrain';

autoenc1.performFcn = 'mse';
autoenc1.performParam.normalization = 'percent';

%autoenc1.performParam.L2WeightRegularization = 0.004;
%autoenc1.performParam.sparsityRegularization = 4;
%autoenc1.performParam.sparsity = 0.15;

%% Train the autoencoder 5-fold cross validation

    m = size(X, 1);
    F = floor(m / K);
    cursor = 0;
    genErrSum = 0;
    while (cursor < F * K)
        starti = cursor + 1;

        if (m - cursor < K)
            endi = m;
        else
            endi = cursor + F;
        end
        Xfold = X(starti:endi, :);
        yfold = T(starti:endi);
        cfold = C(starti:endi);
        Xtfold = X([1:starti - 1 endi + 1:m], :);
        ytfold = T([1:starti - 1 endi + 1:m]);

        autoenc1 = train(autoenc1, Xtfold', Xtfold');
        W1 = autoenc1.IW{1};
        %% REMOVE LAST LAYER
        % Set the size of the input and the 1st layer
        inputSize = size(Xtfold, 2);
        autoencHid1.inputs{1}.size = inputSize;
        autoencHid1.layers{1}.size = hiddenSize;

        % Use the logistic sigmoid transfer function for the first layer
        autoencHid1.layers{1}.transferFcn = 'logsig';

        % Copy the weights and biases from the first layer of the trained
        % autoencoder to this network
        autoencHid1.IW{1,1} = autoenc1.IW{1,1};
        autoencHid1.b{1,1} = autoenc1.b{1,1};

        feat1 = autoencHid1(Xtfold');
        feat1 = feat1';
        %view(autoencHid1);
        %% SUPERVISED TRAINING
        [b,logl,H,stats] = coxphfit(feat1,ytfold);
        
        feat2 = autoencHid1(Xfold');
        feat2 = feat2';
        saeC(hiddenSize) = saeC(hiddenSize) + cIndex(b, feat2, yfold, cfold);
        cursor = cursor + F; 
    end
    saeC(hiddenSize) = saeC(hiddenSize) / K;

%perf = mse(autoenc1, autoenc1(D'), D', 'normalization', 'percent')
end

% net = configure(net, data_tr, data_tr);
% net.trainFcn = 'trainlm';
% net.performFcn = 'mse';
% net.performParam.normalization = 'percent';
% net.trainParam.epochs = 5;
% 
% net = train(net, data_tr, data_tr, 'useParallel', 'yes');
%view(net)
%y = zeros(size(outcome));
%y = net(data_tr);
% 
% 
% perf = mse(net,y,data_tr, 'normalization', 'percent')
% 
% net = configure(net, data_tr, outcome);
% net2 = train(net, data_tr, outcome);
% y2 = net2(data_tr);
% 
% perf2 = mse(net2,y2,outcome, 'normalization', 'percent')