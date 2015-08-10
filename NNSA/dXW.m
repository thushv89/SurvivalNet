function [ tmp ] = dXW( nn, i, l2, l1, p, q )
%DXW calculates dXi/dw for a single sample
% return a vector by [size(Xi, 2) , 1]
        tmp = zeros(size(nn.a{l2}, 2), 1);
        if (l2 == l1 + 1)
            for d = 1: length(tmp)
                if (d == q)
                    tmp(d) = dSigm(nn.a{l1}(i, :) * nn.W{l1}(:, d)) * ...
                    nn.a{l1}(i, p);
                end
            end
        else
            tmp2 = dXW(nn, i, l1 + 1, l1, p, q);
            tmp3 = dXX(nn, i, l2, l1 + 1, q);
            for d = 1: length(tmp)
                tmp(d) = tmp3(d, :) * tmp2(q, :); %should be two scalars

            end
        end
        
end