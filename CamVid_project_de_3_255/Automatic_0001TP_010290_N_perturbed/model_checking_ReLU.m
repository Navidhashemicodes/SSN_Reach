clear

clc


load("Train_data.mat")
load('directions.mat')
load("Train_data2.mat")


load('trained_relu_weights_2h_norm.mat')


dim2 = 720*960*12;
d = 120;

R = zeros(1,100);
R_T = zeros(1,100);
d_hat = zeros(d,100);
d_T = zeros(d,100);
for i=1:100
    loc = floor(60*rand)+1;
    X_rand = X(:, loc);
    % d_hat(:,i) = eval_net1(X_rand, W1, W2, b1 , b2);
    d_hat(:,i) = eval_net2(X_rand, W1, W2, W3, b1 , b2, b3);
    Y_hat =Directions*d_hat(:,i) + C;
    Y_T = Y1(:,:,:,loc);
    Y_T = reshape( Y_T , [dim2 , 1] );
    d_T(:,i) = dYV(:, loc);

    R_T(1,i) = mean(abs(Y_T-Y_hat));
    R(1,i) = mean(abs(d_T(:,i)-d_hat(:,i)));
end

for i=1:d
    figure
    plot(d_hat(i,:), 'blue')
    hold on
    plot(d_T(i,:), 'red')
end


figure

plot(R_T)

figure

plot(R)


function Y = eval_net1(X, W1, W2, b1 , b2)

Y = W2* poslin(W1*X+b1') + b2';

end

function Y = eval_net2(X, W1, W2, W3, b1 , b2, b3)

Y = W3* poslin(W2* poslin(W1*X+b1') + b2' ) + b3';

end