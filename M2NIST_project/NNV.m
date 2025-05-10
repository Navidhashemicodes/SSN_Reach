clear all

clc

close all

%% Load and parse networks into NNV
load('m2nist_dilated_72iou_24layer.mat');

load('m2nist_6484_test_images.mat');


addpath(genpath('C:\Users\navid\Downloads\nnv-cav2021'))


Net = SEGNET.parse(net, 'm2nist_62iou_dilatedcnn_avgpool');


%% create input set
de = 0.004; % size of input

dim1 = 64*84;


ct = 0;
flag = 0;
im = im_data(:,:,1);
at_im = im;
for i=1:64
    for j=1:84
        if im(i,j) > 150
            at_im(i,j) = 0;
            ct = ct + 1;
            if ct == 5
                flag = 1;
                break;
            end
        end
    end
    if flag == 1
        break;
    end
end

dif_im = im - at_im;
noise = -dif_im;
% Perform robustness analysis
V(:,:,:,1) = double(im);
V(:,:,:,2) = double(noise);
C = [1; -1];
d = [1; de-1];
S = ImageStar(V, C, d, 1-de, 1);


c = parcluster('local');
numCores = c.NumWorkers;
poolobj = gcp('nocreate');
delete(poolobj); % reset parpool

tic;
Net.verify(S, {im}, 'approx-star', numCores);

RV = Net.RV;



Lbs = zeros(64,84, 11);
Ubs = zeros(64,84, 11);

parfor i=1:64
    for j=1:84
        for k=1:11
   
            [ Lbs(i,j,k) , Ubs(i,j,k) ] = LP_solver(Net.reachSet.V(i,j,k,1) , Net.reachSet.V(i,j,k,2) ,[],[], 1-de , 1);

        end
    end
end

total_time = toc;

Lb_pixels = reshape(Lbs , [dim1, 11]);
Ub_pixels = reshape(Ubs , [dim1, 11]);
%%%%%%%%%%%%%%



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
SSN = net.predict(im);

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

save("NNV_result_004.mat", "im", "Ub_pixels", "Lb_pixels", "total_time", "RV", "de" , ...
                           "robust", "nonrobust", "unknown", "attacked", "True_class", ...
                           "classes", "noise" );

