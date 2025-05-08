function grad = in_out_der(pre, net, b)


len = length(net.layers);

grad = 1;

for i=1:len
    
    grad = grad * net.weights{end-i+1};
    
    width = length(net.layers{end-i+1});
    grad_diag = sparse(width,1);



    for j=1:width
        if strcmp( net.layers{end-i+1}{j}, 'purelin')
            grad_diag(j,1) = 1;
        elseif strcmp( net.layers{end-i+1}{j}, 'purelin_sat')
            grad_diag(j,1) = 1;
        elseif strcmp( net.layers{end-i+1}{j}, 'poslin')
            grad_diag(j,1) = d_relu(pre{end-i+1}(j,1));
        elseif strcmp( net.layers{end-i+1}{j}, 'softplus')
            grad_diag(j,1) = d_softplus(pre{end-i+1}(j,1), b);
        elseif strcmp( net.layers{end-i+1}{j}, 'swish')
            grad_diag(j,1) = d_swish(pre{end-i+1}(j,1), b);
        elseif strcmp( net.layers{end-i+1}{j}, 'sigmoid')
            grad_diag(j,1) = d_sigmoid(pre{end-i+1}(j,1));
        elseif strcmp( net.layers{end-i+1}{j}, 'tanh')
            grad_diag(j,1) = d_tanh(pre{end-i+1}(j,1));
        else
            error('unknown activation function')
        end
    end



    grad_dia = diag(sparse(grad_diag));

    grad = grad * grad_dia ;

end

grad = sparse(grad * net.weights{1});


end