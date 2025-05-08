function y = swish_vector(x_vec,b)


y = zeros(numel(x_vec),1);
for i = 1:numel(x_vec)
    x = x_vec(i);

    if b*x >50
        y(i) = x;
    elseif abs(b*x)<50
        y(i) = x ./(1+exp(-b*x));
    else
        y(i) = 0;
    end
end


end