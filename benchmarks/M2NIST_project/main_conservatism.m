clear
clc

function [empirical_miscoverage, Y_min, Y_max] = M2NIST_conservatism( start_loc, N_perturbed, image_number, model_name, delta_rgb, Ns , Nsp, Seed0 , LB_out, UB_out )

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

dir = fileparts(dir0);
dir = fileparts(dir);
src_dir = [ dir '/src'];

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
model = Net;
model_source = 'SeriesNetwork';
params = struct;
params.Ns = Ns;
params.Nsp = Nsp;
params.src_dir = src_dir;
params.current_dir = dir0;
params.image_name = image_name;
params.model_name = model_name;
params.model_dir = dir1;
params.image_dir = dir2;
params.delta_rgb = delta_rgb;
params.Seed0 = Seed0;

addpath(genpath(src_dir))
obj = Conservatism_analysis(model,model_source,LB,de,LB_out, UB_out,indices,original_dim, output_dim, params);

[empirical_miscoverage, Y_min, Y_max] = obj.conservatism();

end

model_name = 'm2nist_dilated_72iou_24layer.mat';

Results_eps = cell(1,2);
Results_LB_out = cell(1,2);
Results_UB_out = cell(1,2);


cd('Results_Matlab_main_e')
load('CI_result_ReLU_relaxed_eps_10_Nperturbed_17image_1model_m2nist_dilated_72iou_24layer.mat.mat')
cd ..
LB_out = reshape(Lb_pixels , [],1);
UB_out = reshape(Ub_pixels , [],1);

[empirical_miscoverage, Y_min, Y_max] = M2NIST_conservatism( [1, 1], 17, 1, model_name, 10, 1000000 ,100000, 1000, LB_out, UB_out);
Results_eps{1,1} = empirical_miscoverage;
Results_LB_out{1,1} = Y_min;
Results_UB_out{1,1} = Y_max;




cd('Results_Matlab_main_e')
load('CI_result_ReLU_relaxed_eps_150_Nperturbed_17image_1model_m2nist_dilated_72iou_24layer.mat.mat')
cd ..
LB_out = reshape(Lb_pixels , [],1);
UB_out = reshape(Ub_pixels , [],1);

[empirical_miscoverage, Y_min, Y_max] = M2NIST_conservatism( [1, 1], 17, 1, model_name, 150, 1000000 ,100000, 1000, LB_out, UB_out);
Results_eps{1,2} = empirical_miscoverage;
Results_LB_out{1,2} = Y_min;
Results_UB_out{1,2} = Y_max;


save('Results_conservatism.mat' , "Results_UB_out", "Results_LB_out", "Results_eps")