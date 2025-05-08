function [pre_activation, post_activation] = NN_middle(Net,x,b)
out = x;

the_len = length(Net.layers);
pre_activation = cell(1,the_len);
post_activation = cell(1, the_len);

for i = 1:length(Net.layers)
    pre_activation{i} = Net.weights{i}*out+Net.biases{i};
    len = size(pre_activation{i},1);
    out = sparse(len,1);
    f=Net.layers{i};

    if len<1000
        for ii=1:len
            if strcmp(f(ii), 'poslin')
                out(ii) = poslin(pre_activation{i}(ii));
            elseif strcmp(f(ii), 'softplus')
                out(ii) = Softplus(pre_activation{i}(ii),b);
            elseif strcmp(f(ii), 'swish')
                out(ii) = swish(pre_activation{i}(ii),b);
            elseif strcmp(f(ii), 'sigmoid')
                out(ii) = sigmoid(pre_activation{i}(ii));
            elseif strcmp(f(ii), 'tanh')
                out(ii) = tanh(pre_activation{i}(ii));
            elseif strcmp(f(ii), 'purelin')
                out(ii) = pre_activation{i}(ii);
            elseif strcmp(f(ii), 'purelin_sat')
                out(ii) = pre_activation{i}(ii);
            else
                error('Wrong activation function.')
            end
        end

    else
        parfor ii=1:len
            if strcmp(f(ii), 'poslin')
                out(ii) = poslin(pre_activation{i}(ii));
            elseif strcmp(f(ii), 'softplus')
                out(ii) = Softplus(pre_activation{i}(ii),b);
            elseif strcmp(f(ii), 'swish')
                out(ii) = swish(pre_activation{i}(ii),b);
            elseif strcmp(f(ii), 'sigmoid')
                out(ii) = sigmoid(pre_activation{i}(ii));
            elseif strcmp(f(ii), 'tanh')
                out(ii) = tanh(pre_activation{i}(ii));
            elseif strcmp(f(ii), 'purelin')
                out(ii) = pre_activation{i}(ii);
            elseif strcmp(f(ii), 'purelin_sat')
                out(ii) = pre_activation{i}(ii);
            else
                error('Wrong activation function.')
            end
        end
    end
    post_activation{i} = out;
end


end