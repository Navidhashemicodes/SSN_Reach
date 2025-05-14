% clear
%         clc
% 
%         load('python_data.mat')
%         load('trained_relu_weights_2h_norm.mat')
%         Small_net.weights = {double(W1) , double(W2), double(W3)};
%         Small_net.biases = {double(b1)' , double(b2)', double(b3)'};
% 
%         l1 = size(W1,1);
%         l2 = size(W2,1);
%         l0 = size(W1,2);
% 
%         L = cell(l1,1);
%         L(:) = {'poslin'};
%         Small_net.layers{1} = L ;
%         L = cell(l2,1);
%         L(:) = {'poslin'};
%         Small_net.layers{2} = L ;
% 
%         dim2 = 100;
%         dimp = 3072;
% 
%         tic
%         H.V = [zeros(dim2,1) eye(dim2)];
%         H.C = zeros(1,dim2);
%         H.d = 0;
%         H.predicate_lb = -Conf.';
%         H.predicate_ub =  Conf.';
%         H.dim = dim2;
%         H.nVar= dim2;
%         %%%%%%%%%%%%%%%
%         conformal_time2 = toc;
% 
%         addpath(genpath('C:\Users\navid\Documents\MATLAB\MATLAB_prev\others\Files\CDC2023\Large_DNN\src'))
%         addpath(genpath('C:\Users\navid\Documents\nnv'))
% 
%         tic
%         %%%%%%%%%%%%%%
%         I = Star();
%         I.V = [0.5*ones(l0,1)  eye(l0)];
%         I.C = zeros(1,l0);
%         I.d = 0;
%         I.predicate_lb = -0.5*ones(l0,1);
%         I.predicate_ub =  0.5*ones(l0,1);
%         I.dim =  l0;
%         I.nVar = l0;
% 
% 
%         Principal_reach = ReLUNN_Reachability_starinit(I, Small_net, 'approx-star');
%         Surrogate_reach = affineMap(Principal_reach , VY , C.');
% 
%         %%%%%%%%%%%%%%
%         reachability_time = toc;
%         clear Principal_reach
% 
%         tic
%         %%%%%%%%%%%%%%
%         P_Center = double(Surrogate_reach.V(:,1));
%         P_lb = double([Surrogate_reach.predicate_lb ; H.predicate_lb]);
%         P_ub = double([Surrogate_reach.predicate_ub ; H.predicate_ub]);
% 
% 
%         P_V = [double(Surrogate_reach.V(:,2:end))   double(H.V(:,2:end))];
%         P_Center1 = P_Center(class+1,1) - P_Center;
%         P_V1 = P_V(class+1, :) - P_V;
%         LB = P_Center1 + 0.5*(P_V1+ abs(P_V1))*P_lb + 0.5*(P_V1 - abs(P_V1))*P_ub;
% 
% 
%         %%%%%%%%%%%%%%
%         projection_time = toc;
% 
% 
%         clear  Surrogate_reach
% 
%         save("Matlab_data.mat", 'LB', 'projection_time', 'reachability_time', 'conformal_time2')





clear
        clc

        load('python_data.mat')
        load('trained_A_b.mat')
        load('Bounds.mat')
        A = double(W);
        b = double(b)';

        addpath(genpath('C:\Users\navid\Documents\MATLAB\MATLAB_prev\others\Files\CDC2023\Large_DNN\src'))
        addpath(genpath('C:\Users\navid\Documents\nnv'))

        dim2 = 100;
        dimp = 3072;

        tic
        H.V = [zeros(dim2,1) eye(dim2)];
        H.C = zeros(1,dim2);
        H.d = 0;
        H.predicate_lb = -Conf.';
        H.predicate_ub =  Conf.';
        H.dim = dim2;
        H.nVar= dim2;
        %%%%%%%%%%%%%%%
        conformal_time2 = toc;

        tic
        %%%%%%%%%%%%%%
        I = Star();
        I.V = [0.5*(lb+ub)  eye(dimp)];
        I.C = zeros(1,dimp);
        I.d = 0;
        I.predicate_lb = -0.5*(ub-lb);
        I.predicate_ub =  0.5*(ub-lb);
        I.dim =  dimp;
        I.nVar = dimp;


        Principal_reach = affineMap(I , A , b);
        Surrogate_reach = affineMap(Principal_reach , VY , C.');

        %%%%%%%%%%%%%%
        reachability_time = toc;
        clear Principal_reach

        tic
        %%%%%%%%%%%%%%
        P_Center = double(Surrogate_reach.V(:,1));
        P_lb = double([Surrogate_reach.predicate_lb ; H.predicate_lb]);
        P_ub = double([Surrogate_reach.predicate_ub ; H.predicate_ub]);


        P_V = [double(Surrogate_reach.V(:,2:end))   double(H.V(:,2:end))];
        P_Center1 = P_Center(class+1,1) - P_Center;
        P_V1 = P_V(class+1, :) - P_V;
        LB = P_Center1 + 0.5*(P_V1+ abs(P_V1))*P_lb + 0.5*(P_V1 - abs(P_V1))*P_ub;


        %%%%%%%%%%%%%%
        projection_time = toc;


        clear  Surrogate_reach

        save("Matlab_data.mat", 'LB', 'projection_time', 'reachability_time', 'conformal_time2')