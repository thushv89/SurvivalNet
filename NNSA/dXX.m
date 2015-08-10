function tmp = dXX(nn, X, l2, l1, q)
    tmp = zeros(size(X, 2), 1);
    
    if (l1 == l2 - 1)
        for d = 1: size(X, 2)
            tmp(d) = nn.W{l1}(q, d) * dSigm(nn.a{l1} * nn.W{l1});
        end
    else
        for d = 1: size(X, 2)
            tmp3 =  dXX(nn, X, l1 + 1, l1, q);
            for m = 1: size(nn.a{l1 + 1})
                tmp2 = dXX(nn, X, l2, l1 + 1, m);
 
                tmp(d) = tmp(d) + tmp2(d, :) * ...
                           tmp3(m, :);
            end
        end
    end
end