% function y = NN_eval(NN, X)
%  y = NN.weights{2} * poslin(  NN.weights{1} *  X  + NN.biases{1}  ) +NN.biases{2}; 
% end
function y = NN_eval(NN, X)
 y = NN.weights{3} * poslin( NN.weights{2} * poslin(  NN.weights{1} *  X  + NN.biases{1}  ) +NN.biases{2} )  + NN.biases{3}; 
end