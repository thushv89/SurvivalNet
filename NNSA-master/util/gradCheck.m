function [ diff, grads ] = gradCheck(nn, T, C)
%GRADCHECK gradient cheking wrt w_j in layer l
    e = 1e-4;
    L = nn.n;
    b = nn.W{L - 1};
    diff{L - 1} = zeros(size(b));
    grads{L - 1} = zeros(size(b));
    for j = 1:numel(b)
        gradPlus = LogPartialL(nn.a{L - 1}, T, C, [b(1:j - 1); b(j) + e; b(j+1:end)]);
        gradMinus = LogPartialL(nn.a{L - 1}, T, C, [b(1:j - 1); b(j) - e; b(j+1:end)]);
        approxGrad = (gradPlus -  gradMinus) / (2 * e);
        %diff{L - 1}(j) = nn.deltaW{L - 1}(j)-approxGrad;
        grads{L - 1}(j) = approxGrad;
    end
    for l = 1: L-2
        diff{l} = zeros(size(nn.W{l}));
        grads{l} = zeros(size(nn.W{l}));
        for p = 1:size(nn.W{l}, 1)
            for q = 1:size(nn.W{l}, 2)
                
                nn.W{l}(p, q) = nn.W{l}(p, q) + e;
                nnPlus = mynnff(nn, nn.a{1}(:, 2:end), T, C);
                gradPlus = LogPartialL(nnPlus.a{L - 1}, T, C, b);
                
                nn.W{l}(p, q) = nn.W{l}(p, q) - e - e;
                nnMinus = mynnff(nn, nn.a{1}(:, 2:end), T, C);
                gradMinus = LogPartialL(nnMinus.a{L - 1}, T, C, b);
                
                approxGrad = (gradPlus -  gradMinus) / (2 * e);
                %diff{l}(p, q) = nn.deltaW{l}(p, q)-approxGrad;
                grads{l}(p, q) = approxGrad;
            end
        end
    end
end

