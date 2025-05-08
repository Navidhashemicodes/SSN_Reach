function conf_max = Conf_apply_nostack(Input_Data, Output_Data, net, delta, inv_Coef)

L = length(net);

len = size(Input_Data,2);

NK = size(Output_Data,1);

H = zeros( NK ,  len);

steps = NK/L;
index = 0;
for i = 1:L
    Net = net{i};
    H(index+1:index+steps, :) = NN(Net, Input_Data );
    index = index + steps;
end

loc = floor((len+1)*delta)+1;

if loc>len
    error('Not enough data for Conformal Inference')
end

residual = abs( H - Output_Data);

residual_2 = residual./inv_Coef ;
residual_max  = max(residual_2);
RR = sort(residual_max);

conf_max = RR(1,loc);

end