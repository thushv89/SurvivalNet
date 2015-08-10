prepareData
K = 5;

c = 0;
m = size(X, 1);
F = round(m / K);
cursor = 0;
while (cursor < F * K)
    starti = cursor + 1;
    if (m - cursor < K)
        endi = m;
    else
        endi = cursor + F;
    end
    X_test = X(starti:endi, :);
    T_test = T(starti:endi);
    C_test = C(starti:endi);
    X_train = X([1:starti - 1 endi + 1:m], :);
    T_train = T([1:starti - 1 endi + 1:m]);
    C_train = C([1:starti - 1 endi + 1:m]);
    [b,logl,H,stats] = coxphfit(X_train, T_train);

    c = c + cIndex(b, X_train, T_train, C_train);

    cursor = cursor + F; 
end
c = c / K;