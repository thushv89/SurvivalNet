clear all;
prepareData
StepSize = 1e-3;
hiddenSize = [40 20 20];
%X = X(randperm(size(X, 1), :));
%% train SAE here
%  Setup and train a stacked denoising autoencoder (SDAE)
rng(0, 'v5uniform')
sae = saesetup([size(X, 2) hiddenSize]);

for hl = 1:numel(sae.ae)
    sae.ae{hl}.activation_function       = 'sigm';
    sae.ae{hl}.learningRate              = 1;
    sae.ae{hl}.inputZeroMaskedFraction   = 0;
end

opts.numepochs = 1000;
opts.batchsize = 191;
sae = saetrain(sae, X, opts);
 
%% obtain dimension reduced data
x = X;
for s = 1 : numel(sae.ae)
    t = nnff(sae.ae{s}, x, x);
    x = t.a{2};
    %remove bias term
    x = x(:,2:end);
end
%X = x;    

%% Use the SDAE to initialize a FFNN  
nn = mynnsetup([size(x, 2) hiddenSize 1]);
nn.activation_function              = 'sigm';
nn.learningRate                     = 1;
nn.inputZeroMaskedFraction          = 0;
for hl = 1:nn.n - 2
    nn.W{hl} = sae.ae{hl}.W{1}';
end

%% initialize cox coefficients
[b, logl, H, stats] = coxphfit(x, T);
nn.W{nn.n - 1} = [1;b];

%% feed forward pass
nn = mynnff(nn, X, T, C);

%% calculate lpl and cindex without fine-tuning
Xout = nn.a{nn.n - 1};
Xout = Xout(:, 2:end);
LogPartialL(Xout, T, C, b);
cIndex(b, Xout, T, C)

%% back prop
K = 3;
m = size(X, 1);
F = floor(m / K);
cursor = 0;
maxiter = 350;
lpl_train = zeros(maxiter, 1);
lpl_test = zeros(maxiter, 1);
cindex_train = zeros(maxiter, 1);
cindex_test = zeros(maxiter, 1);
while (cursor < F * K)
    starti = cursor + 1;
    if (m - cursor < K)
        endi = m;
    else
        endi = cursor + F;
    end
    %x_test = reducedX(starti:endi, :);
    x_test = X(starti:endi, :);
    y_test = T(starti:endi);
    c_test = C(starti:endi);
    %x_train = reducedX([1:starti - 1 endi + 1:m], :);
    x_train = X([1:starti - 1 endi + 1:m], :);
    y_train = T([1:starti - 1 endi + 1:m]);
    c_train = C([1:starti - 1 endi + 1:m]);

    nn = mynnsetup([size(x_train, 2) hiddenSize 1]);
    nn.activation_function              = 'sigm';
    nn.learningRate                     = 1;
    nn.inputZeroMaskedFraction          = 0;
    hl = 1;
    for hl = 1:nn.n - 2
        nn.W{hl} = sae.ae{hl}.W{1}';
    end
    
    nn = mynnff(nn, x_train, y_train, c_train);
    Xred_train = nn.a{nn.n - 1};
    Xred_train = Xred_train(:, 2:end);

    %% Train
    iter = 0;
    for iter = 1:1:maxiter
            [diff, grads] = gradCheck(nn, y_train, c_train);

            for j = 1: nn.n - 1
                nn.W{j} = nn.W{j} + StepSize .* grads{j};
            end
            b2 = nn.W{nn.n - 1};
            b2 = b2(2:end, :);
            %nn = mynnff(nn, x_train, y_train, c_train);
            %Xred = nn.a{end - 1};
            %Xred = Xred(:, 2:end);
            iter
            lpl_train(iter) = lpl_train(iter) + LogPartialL(Xred_train, y_train, c_train, b2);
            cindex_train (iter) = cindex_train (iter) + cIndex(b2, Xred_train, y_train, c_train);
            cindex_train_show = cIndex(b2, Xred_train, y_train, c_train)
            
            %% Test
            nn_test = mynnff(nn, x_test, y_test, c_test);
            Xred_test = nn_test.a{end - 1};
            Xred_test = Xred_test(:, 2:end);
            
            cindex_test_show = cIndex(b2, Xred_test, y_test, c_test)
            cindex_test (iter) = cindex_test (iter) + cIndex(b2, Xred_test, y_test, c_test);
            lpl_test(iter) = lpl_test(iter) + LogPartialL(Xred_test, y_test, c_test, b2);
    end
    cursor = cursor + F;
end
cindex_test = cindex_test / K;
cindex_train = cindex_train / K;
lpl_test = lpl_test / K;
lpl_train = lpl_train / K;
   

