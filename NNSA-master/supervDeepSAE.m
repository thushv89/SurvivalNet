%clear all;
prepareData
StepSize = 1e-3;
hiddenSize = [10 10 10 10];

%% train SAE here
%  Setup and train a stacked denoising autoencoder (SDAE)
rng(0, 'v5uniform')
sae = saesetup([size(X, 2) hiddenSize]);

for hl = 1:numel(sae.ae)
    sae.ae{hl}.activation_function       = 'sigm';
    sae.ae{hl}.learningRate              = 1;
    sae.ae{hl}.inputZeroMaskedFraction   = .5;
    sae.ae{hl}.dropoutFraction           = .5;
end

opts.numepochs = 500;
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
%% calculate lpl and cindex without fine-tuning on all data
% Xout = nn.a{nn.n - 1};
% Xout = Xout(:, 2:end);
% LogPartialL(Xout, T, C, b);
% cIndex(b, Xout, T, C)

%% back prop with K-fold cross validation
K = 3;
m = size(X, 1);
F = floor(m / K);
cursor = 0;
maxiter = 150;
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
 
    x_test = X(starti:endi, :);
    y_test = T(starti:endi);
    c_test = C(starti:endi);

    x_train = X([1:starti - 1 endi + 1:m], :);
    y_train = T([1:starti - 1 endi + 1:m]);
    c_train = C([1:starti - 1 endi + 1:m]);

    nn = mynnsetup([size(x_train, 2) hiddenSize 1]);
    nn.activation_function              = 'sigm';
    nn.inputZeroMaskedFraction   = .5;
    nn.dropoutFraction           = .5;
    hl = 1;
    for hl = 1:nn.n - 2
        nn.W{hl} = sae.ae{hl}.W{1}';
    end

    nn = mynnff(nn, x_train, y_train, c_train);
    Xred_train_bias = nn.a{nn.n - 1};
    Xred_train = Xred_train_bias(:, 2:end);
    b = nn.W{nn.n - 1};
    %% calculate lpl and cindex without fine-tuning on training data
    cIndex(b(2:end), Xred_train, y_train, c_train)
    LogPartialL(Xred_train, y_train, c_train, b(2:end))
    %% Train w. bp
    for iter = 1:1:maxiter
            %% change stepsize with iterations
            StepSize = (1e-3) * ((maxiter) - iter) / maxiter + (5e-5)* iter/maxiter;
            if (mod(iter, 10) == 0)
                StepSize
            end
            %%  differentiation
            nn = calcGradient(nn, y_train, c_train, b);
            
            %% gradient checking
            %[diff, grads] = gradCheck(nn, y_train, c_train, b);
            
            %% update weights
            for j = 1: nn.n - 1
                nn.W{j} = nn.W{j} + StepSize .* nn.deltaW{j};
            end
            
            %% get performance with updated weights
            % apply updated parameters to train data
            b2 = nn.W{nn.n - 1};
            b2 = b2(2:end, :);
            nn = mynnff(nn, x_train, y_train, c_train);
            Xred_train_bias = nn.a{end - 1};
            Xred_train = Xred_train_bias(:, 2:end);
      
            lpl_train_show = LogPartialL(Xred_train, y_train, c_train, b2)
            lpl_train(iter) = lpl_train(iter) + lpl_train_show;
            cindex_train_show = cIndex(b2, Xred_train, y_train, c_train)
            cindex_train (iter) = cindex_train (iter) + cindex_train_show;
           
            %hold on
            
            %% Test
            % apply updated parameters to test data
            nn_test = mynnff(nn, x_test, y_test, c_test);
            Xred_test = nn_test.a{end - 1};
            Xred_test = Xred_test(:, 2:end);
            
            cindex_test_show = cIndex(b2, Xred_test, y_test, c_test)
            cindex_test (iter) = cindex_test (iter) + cindex_test_show;
            lpl_test_show = LogPartialL(Xred_test, y_test, c_test, b2) 
            lpl_test(iter) = lpl_test(iter) + lpl_test_show;
            iter
    end
    cursor = cursor + F;
 end
cindex_test = cindex_test / K;
cindex_train = cindex_train / K;
lpl_test = lpl_test / K;
lpl_train = lpl_train / K;

plot(1:iter, cindex_train(1: iter));
plot(1:iter, lpl_train(1: iter));

plot(1:iter, cindex_test(1: iter));
plot(1:iter, lpl_test(1: iter));

