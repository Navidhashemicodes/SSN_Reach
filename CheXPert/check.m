clear
clc

load('python_data.mat')
load('trained_relu_weights_2h_norm.mat')
Small_net.weights = {double(W1) , double(W2), double(W3)};
Small_net.biases = {double(b1)' , double(b2)', double(b3)'};

l1 = size(W1,1);
l2 = size(W2,1);
l0 = size(W1,2);

L = cell(l1,1);
L(:) = {'poslin'};
Small_net.layers{1} = L ;
L = cell(l2,1);
L(:) = {'poslin'};
Small_net.layers{2} = L ;

dim2 = 512*512*1;
dimp = 512*512;

tic
H.V = [sparse(dim2,1) speye(dim2)];
H.C = sparse(1,dim2);
H.d = 0;
H.predicate_lb = -Conf.';
H.predicate_ub =  Conf.';
H.dim = dim2;
H.nVar= dim2;
%%%%%%%%%%%%%%%
conformal_time2 = toc;

addpath(genpath('C:\\Users\\navid\\Documents\\MATLAB\\MATLAB_prev\\others\\Files\\CDC2023\\Large_DNN\\src'))

tic
%%%%%%%%%%%%%%
I = Star();
I.V = [0.5*ones(l0,1)  eye(l0)];
I.C = zeros(1,l0);
I.d = 0;
I.predicate_lb = -0.5*ones(l0,1);
I.predicate_ub =  0.5*ones(l0,1);
I.dim =  l0;
I.nVar = l0;


Principal_reach = ReLUNN_Reachability_starinit(I, Small_net, 'approx-star');
Surrogate_reach = affineMap(Principal_reach , Directions.' , C.');

%%%%%%%%%%%%%%
reachability_time = toc;
clear Principal_reach

tic
%%%%%%%%%%%%%%
P_Center = sparse(double(Surrogate_reach.V(:,1)));
P_lb = double([Surrogate_reach.predicate_lb ; H.predicate_lb]);
P_ub = double([Surrogate_reach.predicate_ub ; H.predicate_ub]);


P_V = [double(Surrogate_reach.V(:,2:end))   double(H.V(:,2:end))];
Lb = P_Center + 0.5*(P_V + abs(P_V))*P_lb + 0.5*(P_V - abs(P_V))*P_ub;
Ub = P_Center + 0.5*(P_V + abs(P_V))*P_ub + 0.5*(P_V - abs(P_V))*P_lb;


Lb_pixels = reshape(Lb , [dimp, 1]);
Ub_pixels = reshape(Ub , [dimp, 1]);
%%%%%%%%%%%%%%
projection_time = toc;


clear  Surrogate_reach

save("Matlab_data.mat", 'Lb_pixels', 'Ub_pixels', 'projection_time', 'reachability_time', 'conformal_time2')




for j=1:512
    for i = 1:512
        t = (j-1)*512+i;
        if Lb_pixels(t,1) > 0
            class_members = 1;
        elseif Ub_pixels(t,1) <=0
            class_members = 0;
        else
            class_members = [0, 1];
        end
        classes{i,j} = class_members;
    end
end



params = importONNXFunction('lung_segmentation.onnx', 'lungSegFunc');


img = imread('CHNCXR_0005_0.png');
img = double(img) / 255;
img = imresize(img, [512, 512]);
img = reshape(img, [512, 512, 1, 1]);


[UNet, ~] = lungSegFunc(img, params, 'InputDataPermutation', 'auto');


True_class = cell(512, 512);
for i =1:512
    for j=1:512

        if UNet(i,j) > 0
            True_class{i,j} = 1;
        else
            True_class{i,j} = 0;
        end
    end
end


robust = 0 ;
nonrobust = 0;
unknown = 0;
attacked = 5;

for i =1:512
    for j=1:512
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



RV = 100* robust / (512*512);
RS = (nonrobust + unknown) / attacked;