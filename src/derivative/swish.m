% function y = swish(x,b)
% 
% y = x ./(1+exp(-b*x));
% 
% end
% 
% This function is inspired by NN, NN_middle and in_out_der it only accept
% scalar not vector
function y = swish(x,b)

if numel(x)>1
    error('vector is not allowed.')
end

if b*x >50
    y = x;
elseif abs(b*x)<50
    y = x ./(1+exp(-b*x));
else
    y = 0;
end


end