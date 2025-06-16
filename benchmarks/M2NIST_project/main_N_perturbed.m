clear
clc

function verify_M2NIST_on_CNNS( start_loc, N_perturbed, image_number, model_name, delta_rgb, N, N_dir, Ns , rank, guarantee  )

Nsp = Ns;
height = 64;
width = 84;
n_class = 11;
n_channel = 1;
std_vals = [1,1,1];
de = delta_rgb./std_vals;
dir0 = pwd;
dir1 = [ dir0  '/models'];
dir2 = [ dir0  '/images'];
cd(dir1)
load(model_name)
layers = net.Layers;
newLayers = layers(1:end-2);
Net = SeriesNetwork(newLayers);

cd(dir2)
load("m2nist_6484_test_images.mat")
img = im_data(:,:,image_number);
image_name = num2str(image_number);

cd(dir0)
SSN = net.predict(img);
True_class = cell(height, width);
for i =1:height
    for j=1:width
        [~, True_class{i,j}] = max(SSN(i,j,:));
    end
end
class_threshold = [];
dir = fileparts(dir0);
dir = fileparts(dir);
src_dir = [ dir '/src'];
nnv_dir = '/home/hashemn/nnv';
ct = 0;
flag = 0;
at_im = img;
indices = zeros(N_perturbed , 2);
for i=start_loc(1):height
    for j=start_loc(2):width
        if img(i,j) > 150
            at_im(i,j) = 0;
            ct = ct + 1;
            indices(ct , :) = [i,j];
            disp([i,j])
            if ct == N_perturbed
                flag = 1;
                break;
            end
        end
    end
    if flag == 1
        disp([num2str(N_perturbed) ' pixels found.'])
        break;
    end
end

LB = at_im;
original_dim = [n_channel, height, width];
output_dim = [n_class, height, width];
mode = 'relu';
model = Net;
model_source = 'SeriesNetwork';
params = struct;
params.Nt = N;
params.N_dir = N_dir;
params.Ns = Ns;
params.Nsp = Nsp;
params.src_dir = src_dir;
params.current_dir = dir0;
params.image_name = image_name;
params.model_name = model_name;
params.model_dir = dir1;
params.image_dir = dir2;
params.trn_batch = floor(N_dir/3);
% if strcmp(model_source , 'onnx')
%     params.model_params = model_params;
% end
params.num_sample_fittable_in_2GB = N_dir;
params.dims = [-1 -1];
params.threshold_normal = 1e-5;
params.guarantee = guarantee;
params.rank= rank;
params.epochs = 200;
params.delta_rgb = delta_rgb;

addpath(genpath(src_dir))
addpath(genpath(nnv_dir))
obj = verify_NN(True_class,class_threshold,model,model_source,LB,de,indices,original_dim,output_dim,mode, params);

[~, ~] = obj.Mask_titles();

end



for i=1:20
    for j=1:6
        if i~=13 || j~=6
            model_name = 'm2nist_dilated_72iou_24layer.mat';
            Text = sprintf("verify_M2NIST_on_CNNS( [1, 1], %d, %d, '%s', 3, 2000, 2000, 100000 , 99999, 0.9999  )",j*17,i, model_name);
            eval(Text);
        end
    end
end

for i=1:20
    for j=1:6
        if i~=13 || j~=6
            model_name = 'm2nist_75iou_transposedcnn_avgpool.mat';
            Text = sprintf("verify_M2NIST_on_CNNS( [1 , 1], %d, %d, '%s', 3, 2000, 2000, 100000 , 99999, 0.9999  )",j*17,i, model_name);
            eval(Text)
        end
    end
end


for i=1:20
    for j=1:6
        if i~=13 || j~=6
            model_name = 'm2nist_62iou_dilatedcnn_avgpool.mat';
            Text = sprintf("verify_M2NIST_on_CNNS( [1 , 1], %d, %d, '%s', 3, 2000, 2000, 100000 , 99999, 0.9999  )",j*17,i, model_name);
            eval(Text)
        end
    end
end

