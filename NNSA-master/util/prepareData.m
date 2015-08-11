load AML.mat
% load('Ximputed.mat');
% % load('Ximputed_t.mat');
% %training data, survival time and censoring
% D = Ximputed;
% X = [D(:,2:18), D(:,23:end)];
% %X = normalizeClns(X, [2,3,18:271]);
% C = D(:, 20);
% %C = normalizeClns(C, 1);
% T = D(:, 21);
% %T = normalizeClns(T, 1);
N = length(T);
X = (X - ones(N,1)*mean(X,1)) ./ (ones(N,1)*std(X,[],1));
randrows = randperm(size(X, 1));
T = T(randrows);
C = C(randrows);
X = X(randrows, :);
%% get rid of protein data
%X = X(:, 1:20);

% %% randomize data
%  X = rand(191, 271);
%  T = rand(191, 1);
%  C = (rand(191, 1) > .5);