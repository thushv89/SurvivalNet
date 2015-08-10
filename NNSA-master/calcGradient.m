function [ nn ] = calcGradient( nn, X, Y, C, b )
% calculate dL/dw given all samples in layer n - 1
    J = nn.n; % number of layers
    m = size(X, 1);
    dLPLdX = dLogPartialLdX(X, Y, C, b);
    nn.deltaW{J - 1} = dLogPartialL(X, Y, C, b)';
    for j = (J - 2):-1:1
        [P, Q] = size(nn.W{j});
        nn.deltaW{j} = zeros(P, Q);
        for p = P:-1:1       
            for q = Q:-1:1
                % for each w, dL/dw, note that L is the result of summation
                % over Xi
                for i = 1:1:m
                    % calculate derivative
                    nn.deltaW{j}(p, q) = nn.deltaW{j}(p, q) + dLPLdX(i, :) * ...
                        dXW(nn, i, J - 1, j, p, q);
                end
            end
        end
    end
end