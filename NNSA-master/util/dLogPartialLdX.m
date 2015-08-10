function [ dLPLdX ] = dLogPartialLdX( X, Y, C, b )
% given n samples in n rows of X, computes the gradient of the log partial likelihood of a cox model
% with respect to n samples and returns the result in dLPLdX
    [Ysorted, Order] = sort(Y);
    %Censorted = Censored(Order);
    Xsorted = X(Order,:);
    [~, atRiskBegin] = ismember(Ysorted, Ysorted);
    
    
    m = size( X, 1);
    n = size( X, 2);
    
    denom = zeros(m , 1);
    dLPLdX = repmat(C, 1, n) .* (zeros(m , length(b)) + repmat(b', m , 1)) ;
    for i = 1:m 
        denom(i) = sum(exp(Xsorted(atRiskBegin(i):end, :) * b));
    end
    
    for i = 1:m % iterate to calc derivative wrt to all samples
        for j = 1:m % the sum in the ll expression
            if (~C(j) && (Order(j) <= Order(i)))
                dLPLdX(i, :) = dLPLdX(i, :) - ...
                 (b' .* exp(X(i, :) * b) ./ denom(j));         
            end
        end
%        dLPLdX(i, :) = rehsape(dLPLdX, [1, length(b)]);
    end
    
end

