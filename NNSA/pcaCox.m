prepareData
%% constants 
iter = 1;
K = 5;


c = zeros(iter, 1);
%% K-fold cross-validation
m = size(X, 1);
F = floor(m / K);   
for i = 1:iter
    i
    cursor = 0;
    %% pca on whole data
    [coeff,score] = pca(X, 'NumComponents', 10 + 18);
    while (cursor < F * K)
        starti = cursor + 1
        if (m - cursor < K)
            endi = m
        else
            endi = cursor + F
        end
        Xfold = score(starti:endi, :);
        yfold = T(starti:endi);
        cfold = C(starti:endi);
        Xtfold = score([1:starti - 1 endi + 1:m], :);
        ytfold = T([1:starti - 1 endi + 1:m]);
        
        %% regression on training set 
        [b,logl,H,stats] = coxphfit(Xtfold, ytfold);
        
        %% cindex calculation on testing set
        cIndex(b, Xfold, yfold, cfold)
        c(i) = c(i) + cIndex(b, Xfold, yfold, cfold);
        
        cursor = cursor + F;
    end
    c(i) = c(i) / K;
end