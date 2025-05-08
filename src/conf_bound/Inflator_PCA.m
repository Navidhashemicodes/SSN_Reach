function [Hs,  Res_trn_times, Inflating_time]  = Inflator_PCA(Test_data, Train_data, net, delta, n)

len = size(Test_data,2);


loc = floor((len+1)*delta)+1;

if loc>len
    error('Not enough data for Conformal Inference')
end

L = length(net);

NK = size(Test_data,1) - n;
steps = NK/L;

R = zeros(L,len);
V = cell(1, L);
Mean = cell(1, L);
scaler = cell(1, L);

Res_trn_times = cell(1, L);
Inflating_times = zeros(1,L);

for i = 1:L

    disp(['Conformal analysis on model' num2str(i-1)])
    
    index = (i-1)*steps;

    Net = net{i};

    tic

    H_trn = NN(Net, Train_data(1:n , :) );
    

    errors =  Train_data(n+index+1:n+index+steps , :) - H_trn;
    M = cov(errors');
    [V{i} , ~] = eig(M);
    Mean{i} = mean(errors, 2);
    Errors = abs(V{i}'*(errors - Mean{i}));
    scaler{i} = max(Errors , [] , 2);
    
    Res_trn_times{i} = toc;

    tic

    H = NN(Net, Test_data(1:n , :) );
    residuals =  Test_data(n+index+1:n+index+steps , :) - H;
    Residuals = V{i}' * (residuals - Mean{i});
    R(i,:) = max(abs(Residuals)./scaler{i});
    
    Inflating_times(i) = toc;


end

tic

Rs = max(R);

RR = sort(Rs);

conf_max = RR(1,loc);

Inflating_time = toc + sum(Inflating_times);

Hs = cell(1,L);


%%% The predicates are nothing but the residual errors on the defined
%%% coordinates which is set on the mean and alligned on the V. Consider V'
%%% is equal to inv(V) as M is a PSD matrix.

for i=1:L

    Conf_overall = conf_max*scaler{i};
    Hs{i} = Star();
    Hs{i}.V = [Mean{i}, V{i}];
    Hs{i}.C = zeros(1, steps);
    Hs{i}.d = 0;
    Hs{i}.predicate_lb = -Conf_overall;
    Hs{i}.predicate_ub =  Conf_overall;
    Hs{i}.dim = steps;
    Hs{i}.nVar = steps;

end

end