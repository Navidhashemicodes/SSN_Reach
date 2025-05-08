function [ C_max, inv_Coefficients ] = Analytic_LP(inv_Coefficients, Input_Data , Output_Data, net)

H = NN(net, Input_Data );
residual = abs( H - Output_Data);

residual_2 = residual./inv_Coefficients;
C  = max(residual_2);

C_max = max(C);

Residual = residual./C;
inv_Coefficients = max(Residual, [] , 2);

end