function [ y ] = dSigm( x )
    y = sigm(x) .* (ones(size(x)) - sigm(x));
end

