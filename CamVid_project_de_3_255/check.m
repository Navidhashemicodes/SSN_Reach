clear all

close all

clc
% Set your directories

dir0 = pwd;

dir1 = 'C:\Users\navid\Documents\MATLAB\MATLAB_prev\others\Files\CDC2023\Large_DNN\Case_study\CamVid\Pytorch_repo\BiSeNet-master\model';     % <-- directory where BiSeNet.onnx is saved
dir2 = 'C:\Users\navid\Documents\MATLAB\MATLAB_prev\others\Files\CDC2023\Large_DNN\Case_study\CamVid\Pytorch_repo\CamVid\test';       % <-- directory where your CamVid image is

% Step 1: Load the ONNX model as a function
cd(dir1)
params = importONNXFunction('BiSeNet.onnx', 'BiSeNetFunc');

% Step 2: Load and preprocess a CamVid RGB image
cd(dir2)
img = imread('0006R0_f02190.png');  % <-- replace with a valid CamVid image filename

img_double = im2double(img);  % Now range is [0, 1], size is still H x W x 3

% Step 2: Normalize channels separately
mean_vals = [0.485, 0.456, 0.406];
std_vals  = [0.229, 0.224, 0.225];

% Preallocate output
img_norm = zeros(size(img_double));

% Normalize each channel
for c = 1:3
    img_norm(:,:,c) = (img_double(:,:,c) - mean_vals(c)) / std_vals(c);
end

% Step 4: Call the ONNX function model
cd(dir1)
% [output1, ~] = BiSeNetFunc(img_norm, params, 'InputDataPermutation', 'auto');

img_norm = single(img_norm);


[output2, ~] = BiSeNetFunc(img_norm, params, 'InputDataPermutation', 'auto');


% output1 = reshape(output1 , [numel(output1) , 1]);
% output2 = reshape(output2 , [numel(output2) , 1]);
% plot(output1 - output2)

cd(dir0)
% figure
% [~ , Seg] = max(output1, [], 3);
% visualize(Seg)
% 
figure
[~ , Seg] = max(output2, [], 3);
visualize(Seg)
