clear all
clc
close all

addpath(genpath('C:\Program Files\Mosek'));
addpath(genpath('C:\Users\navid\Documents\MATLAB\MATLAB_prev\others\Files\CDC2023\Large_DNN\src'))


N = 20000; ell = 19993;  %%%  CI  ==>  Pr[ Pr[ membership ] > 99.99% ] > 99.99% 
epsilon = 0.001;  %%% failure probability of the reachset whic is a function of N and ell. if it is near to zero we take cube center, if it is near to 0.05 we take the mean value 
de = 0.004;
le = 1-de;
ue = 1;

%%%%%%%%%%%%%%%%%%%%%%%
load("trained_relu_weights_norm.mat")
small_net.weights = {double(W1) , double(W2)};
small_net.biases = {double(b1)' , double(b2)'};
L = cell(10,1);
L(:) = {'poslin'};
small_net.layers{1} = L ;
%%%%%%%%%%%%%%%%%%%%%%%%




dim1 = 64*84;
dim2 = 64*84*11;

load("Train_data.mat")
Nt = size(X,2);
Y = cat(4 ,Y1 ,Y2 ,Y3 ,Y4 ,Y5);
Y = reshape( Y , [dim2 , Nt] );

tic
%%%%%%%%%%%%%%%
res_trn = abs( Y - ( directions * NN_eval(small_net , X)  + C ) );
threshold_normal = 10^(-15);
res_max = max( res_trn ,[] ,2 );
indices = find(res_max < threshold_normal);
res_max(indices,1) = threshold_normal;
%%%%%%%%%%%%%%%
trn_time1 = toc;

clear Y X  res_trn Y1 Y2 Y3 Y4 Y5 

load("Test_data1.mat")
Ns = size(X_test1,2);
Y_test  = cat(4, Y_test11 ,Y_test12 ,Y_test13 ,...
                 Y_test14 ,Y_test15 ,Y_test16 ,...
                 Y_test17 ,Y_test18 ,Y_test19 ,...
                 Y_test110 );
X_test  = X_test1;
clear X_test1

Y_test = reshape( Y_test , [dim2 , Ns] );

tic
%%%%%%%%%%%%%%
res_tst = abs(Y_test - ( directions * NN_eval( small_net , X_test) + C ));
Rs = max( res_tst ./ res_max);
Rs_sorted = sort(Rs);
R_star = Rs_sorted(:,ell);

Conf = R_star*res_max;

H = Star();
H.V = [zeros(dim2,1) eye(dim2)];
H.C = zeros(1,dim2);
H.d = 0;
H.predicate_lb = -Conf;
H.predicate_ub =  Conf;
H.dim = dim2;
H.nVar= dim2;
%%%%%%%%%%%%%%%
conformal_time = toc;

clear Y_test Y_test11 Y_test12 Y_test13 Y_test14 Y_test15 Y_test16 Y_test17 Y_test18 Y_test19 Y_test110 
clear X_test rest_tst



tic
%%%%%%%%%%%%%%
I = Star();
I.V = [0.5  0.5];
I.C = zeros(1,1);
I.d = 0;
I.predicate_lb = -1;
I.predicate_ub =  1;
I.dim =  1;
I.nVar = 1;

Principal_reach = ReLUNN_Reachability_starinit(I, small_net, 'approx-star');
Surrogate_reach = affineMap(Principal_reach , directions , C);

%%%%%%%%%%%%%%
% Probabiltic_reach = Sum( Surrogate_reach , H );
P_Center = sparse(Surrogate_reach.V(:,1));
P_V = [sparse(Surrogate_reach.V(:,2:end))   sparse(H.V(:,2:end))];
P_C = blkdiag(sparse(Surrogate_reach.C) , sparse(H.C));
P_d = [ sparse(Surrogate_reach.d); sparse(H.d) ];
P_lb = [Surrogate_reach.predicate_lb ; H.predicate_lb];
P_ub = [Surrogate_reach.predicate_ub ; H.predicate_ub];
%%%%%%%%%%%%%%%
reachability_time = toc;

clear Principal_reach  Surrogate_reach

tic
%%%%%%%%%%%%%%%
[ Lb , Ub ] = LP_solver(P_Center , P_V ,P_C , P_d  ,P_lb ,P_ub);

Lb_pixels = reshape(Lb , [dim1, 11]);
Ub_pixels = reshape(Ub , [dim1, 11]);
%%%%%%%%%%%%%%
projection_time = toc;


classes = cell(64,84);

tic

for j=1:84
    for i = 1:64
        t = (j-1)*64+i;
        [L_star, my_class] = max(Lb_pixels(t,:));
        class_members = [];
        for k = 1:11
            if L_star <= Ub_pixels(t,k)
                class_members = [class_members , k];
            end
        end
        classes{i,j} = class_members;
    end
end


load("m2nist_dilated_72iou_24layer.mat")
SSN = net.predict(Input);

True_class = cell(64, 84);
for i =1:64
    for j=1:84
       [~, True_class{i,j}] = max(SSN(i,j,:));
    end
end


robust = 0 ;
nonrobust = 0;
unknown = 0;
attacked = 5;

for i =1:64
    for j=1:84
        if (length(classes{i,j}) == 1)
            if classes{i,j} == True_class{i,j}
                robust = robust + 1;
            else 
                nonrobust = nonrobust + 1;
            end
        else
            if ismember(True_class{i,j} , classes{i,j} )
                unknown = unknown + 1;
            else
                nonrobust = nonrobust + 1;
            end
        end
    end
end


RV = 100* robust / dim1;
RS = (nonrobust + unknown) / attacked;

disp(nonrobust)


overall_time = trn_time1 + conformal_time + reachability_time + projection_time + test_data_run1 + train_data_run;

save("CI_result_middle_guarantee_ReLU_2e4_004.mat", "robust", "nonrobust", "attacked", "unknown", "True_class", "classes", "Conf"        , ...
                                                     "N", "de", "ell", "epsilon", "Lb_pixels", "Ub_pixels", "Ns", "Nt", "R_star"         , ...
                                                     "res_max", "RV", "threshold_normal" , "overall_time", "trn_time1" , "conformal_time", ...
                                                     "reachability_time" , "projection_time" , "test_data_run1" , "train_data_run" );


clear all




