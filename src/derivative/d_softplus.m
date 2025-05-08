% function y = d_softplus(x,b)
% 
% y = exp(b*x) ./ ( 1 + exp(b*x) );
% 
% end

% This function is inspired by NN, NN_middle and in_out_der it only accept
% scalar not vector
function y = d_softplus(x,b)

if numel(x)>1
    error('vector is not allowed.')
end

if b*x>50
    y = 1;
elseif abs(b*x)<50
    y = exp(b*x) ./ ( 1 + exp(b*x) );
else
    y = 0;
end

end