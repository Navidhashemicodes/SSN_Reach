clear all 

clc

close all


load("m2nist_dilated_72iou_24layer.mat")
layers = net.Layers;  
newLayers = layers(1:end-2);
Net = SeriesNetwork(newLayers);

save("Logits_Net.mat", "Net")

load("m2nist_6484_test_images.mat")


Input = gpuArray(im_data(:,:,1));


de = 0.004;
lalpha = 1-de;
ualpha = 1;


ct = 0;
flag = 0;
at_im = Input;
for i=1:64
    for j=1:84
        if Input(i,j) > 150
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

dif_im = Input - at_im;
noise = -dif_im;

N = 10000;
db = ualpha - lalpha;
Y = zeros(64,84,11,N);
X = zeros(1,N);
tic
%%%%%%%%%%%%%%
for i=1:N
    disp(i)
    Rand = rand;
    alpha = lalpha + Rand*db;
    Inp = Input + alpha*noise;
    X(1,i) = Rand;
    Y(:,:,:,i) = Net.predict(Inp);
end
%%%%%%%%%%%%%
train_data_run_1 = toc;

ind = 0;
Y1 = Y(:,:,:,ind+1:ind+2000);
ind = ind+2000;
Y2 = Y(:,:,:,ind+1:ind+2000);
ind = ind+2000;
Y3 = Y(:,:,:,ind+1:ind+2000);
ind = ind+2000;
Y4 = Y(:,:,:,ind+1:ind+2000);
ind = ind+2000;
Y5 = Y(:,:,:,ind+1:ind+2000);

tic
%%%%%%%%%%%%%%%%
epsilon = 0.001;
n1 = numel(Y(:,:,:,1));
Y = reshape(Y, [n1 , N]);

if epsilon < 0.05
    C = 20* (  epsilon*(mean(Y,2))   +  (0.05-epsilon) * 0.5*( min(Y , [], 2) + max(Y , [] , 2) ));
else
    C = mean(Y,2);
end

dY = Y - C;

load('directions.mat')

d = 10;

directions = double(Directions(:, 1:d));


dYV  = directions' * (Y - C );
%%%%%%%%%%%%%%%%%%%
train_data_run_2 = toc;

train_data_run = train_data_run_1 + train_data_run_2;
Input = gather(Input);
noise = gather(noise);
save("Train_data.mat", "dYV", "X", "C", "directions", "noise", "Input", ...
                       "Y1", "Y2", "Y3", "Y4", "Y5" , "train_data_run");

clear Y 
clear X

N = 20000;
db = ualpha - lalpha;
Y_test1 = zeros(64,84,11,N);
X_test1 = zeros(1,N);
tic
%%%%%%%%%%%%%%%%%%%%%
for i=1:N
    disp(['part 1 number ' num2str(i)])
    Rand = rand;
    alpha = lalpha + Rand*db;
    Inp = Input + alpha*noise;
    Y_test1(:,:,:,i) = Net.predict(Inp);
    X_test1(1,i) = Rand;
end
%%%%%%%%%%%%%%%%%%%%
test_data_run1 = toc;

ind = 0;
X_test11 = X_test1(1,ind+1:ind+2000);
Y_test11 = Y_test1(:,:,:,ind+1:ind+2000);
ind = ind+2000;
X_test12 = X_test1(1,ind+1:ind+2000);
Y_test12 = Y_test1(:,:,:,ind+1:ind+2000);
ind = ind+2000;
X_test13 = X_test1(1,ind+1:ind+2000);
Y_test13 = Y_test1(:,:,:,ind+1:ind+2000);
ind = ind+2000;
X_test14 = X_test1(1,ind+1:ind+2000);
Y_test14 = Y_test1(:,:,:,ind+1:ind+2000);
ind = ind+2000;
X_test15 = X_test1(1,ind+1:ind+2000);
Y_test15 = Y_test1(:,:,:,ind+1:ind+2000);
ind = ind+2000;
X_test16 = X_test1(1,ind+1:ind+2000);
Y_test16 = Y_test1(:,:,:,ind+1:ind+2000);
ind = ind+2000;
X_test17 = X_test1(1,ind+1:ind+2000);
Y_test17 = Y_test1(:,:,:,ind+1:ind+2000);
ind = ind+2000;
X_test18 = X_test1(1,ind+1:ind+2000);
Y_test18 = Y_test1(:,:,:,ind+1:ind+2000);
ind = ind+2000;
X_test19 = X_test1(1,ind+1:ind+2000);
Y_test19 = Y_test1(:,:,:,ind+1:ind+2000);
ind = ind+2000;
X_test110 = X_test1(1,ind+1:ind+2000);
Y_test110 = Y_test1(:,:,:,ind+1:ind+2000);


save("Test_data1.mat", "Y_test11", "Y_test12", "Y_test13", "Y_test14", "Y_test15",...
     "Y_test16", "Y_test17", "Y_test18", "Y_test19", "Y_test110", ...
     "X_test11", "X_test12", "X_test13", "X_test14", "X_test15",...
     "X_test16", "X_test17", "X_test18", "X_test19", "X_test110", "test_data_run1");

clear Y_test1 Y_test11 Y_test12 Y_test13 Y_test14 Y_test15 Y_test16 Y_test17 Y_test18 Y_test19 Y_test110 
clear X_test1 X_test11 X_test12 X_test13 X_test14 X_test15 X_test16 X_test17 X_test18 X_test19 X_test110 



