function verify_CamVid_on_BiSeNet( start_loc, N_perturbed, image_name, delta_rgb, N, N_dir, Ns , rank, guarantee  )


if N>2100
    error('CamVid samples are very large, considering memory constrained please set N less than 2100.')
end


dir0 = pwd;
dir = fileparts(dir0);

dir1 = [ dir  '/Pytorch_repo/BiSeNet-master/model'];
dir2 = [ dir  '/Pytorch_repo/CamVid/test'];
% Load the ONNX model
cd(dir1)
params = importONNXFunction('BiSeNet.onnx', 'BiSeNetFunc');


cd(dir2)
img = imread(image_name);
% img = imread('0006R0_f02190.png');

img_double = im2double(img);  % Now range is [0, 1], size is still H x W x 3


ct = 0;
flag = 0;
at_im = img_double;
indices = zeros(N_perturbed , 2);
for i=start_loc(1):720
    for j=start_loc(2):960
        if min(img_double(i,j,:) ) > 150/255
            at_im(i,j,:) = 0;
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




% Step 2: Normalize channels separately
mean_vals = [0.485, 0.456, 0.406];
std_vals  = [0.229, 0.224, 0.225];

% Preallocate output
at_im_norm = zeros(size(img_double));

% Normalize each channel
for c = 1:3
    at_im_norm(:,:,c) = (at_im(:,:,c) - mean_vals(c)) / std_vals(c);
    im_norm = (img_double(:,:,c) - mean_vals(c)) / std_vals(c);
end


% Input = gpuArray(at_im_norm);
Input = at_im_norm;


de = delta_rgb/255;
rng(0)

Y = zeros(720,960,12,N);
X = zeros(3*N_perturbed , N);
cd(dir1)
tic
%%%%%%%%%%%%%%
parfor i=1:N
    disp(i)
    Rand = rand(3*N_perturbed,1);
    Rand_matrix = mat_generator_no_third(Rand , indices , [720,960,3]);
    d_at = zeros(720,960,3);
    for c=1:3
        d_at(:,:,c) = de * Rand_matrix(:,:,c) / std_vals(c) ;
    end
    Inp = single(Input + d_at);
    X(:,i) = Rand;
    [Y(:,:,:,i), ~] = BiSeNetFunc(Inp, params, 'InputDataPermutation', 'auto');
end
%%%%%%%%%%%%%
train_data_run_1 = toc;


Input_cpu = gather(Input);
bench_cpu = gather(im_norm);

if N_dir >30
    leng = 30;
else
    leng = N_dir;
end
ind = 0;
if mod(N_dir , leng) == 0
    last_i = N_dir/leng ;
else
    error('Considering Camvid_dataset, if N_dir is more than 30 it should be devisible by 30.')
end
for i=1:last_i
    eval(['Y' num2str(i) ' = Y(:,:,:,ind+1:ind+leng);' ]);
    ind = ind+leng;
end

cd(dir0)
Text = ' save("Direction_data.mat" ';
for i=1:last_i
    Text = [ Text  ' ,"Y' num2str(i) '" ']; %#ok<*AGROW>
end
Text = [ Text ');'];
eval(Text);

Text = 'clear Y1'; %%% You always need to have something here even if it is nonsense, otherwise it clears everything
for i=1:last_i
    Text = [ Text  ' Y' num2str(i) '  '];
end
eval(Text);



cd(dir0)
mat_file_path = [dir0 '/Direction_data.mat'];

command = sprintf(['python Direction_trainer.py --mat_file_path "%s" ' ...
                   '--num_files %d --N_dir %d '], ...
                   mat_file_path,  last_i, N_dir);

status = system(command);
% system('python Direction_trainer.py')


n1 = numel(Y(:,:,:,1));
Y = reshape(Y, [n1 , N]);

C = 20* (  0.01*(mean(Y,2))   +  (0.05-0.01) * 0.5*( min(Y , [], 2) + max(Y , [] , 2) ));

% load('directions.mat')

pyenv
npz = py.numpy.load('directions.npz');
shape = cellfun(@double, cell(npz{'Directions'}.shape));
Directions_py = py.numpy.array(npz{'Directions'});
Directions = single(Directions_py.reshape(int64(shape)));  % use reshape to preserve dimensions
Direction_Training_time = double(npz{'Direction_Training_time'});
clear Directions_py  npz


dYV  = Directions' * (Y - C );

save("Reduced_dimension.mat", "dYV", "X");

cd(dir0)
system('python Trainer_ReLU.py')


load("trained_relu_weights_2h_norm.mat")
small_net.weights = {double(W1) , double(W2), double(W3)};
small_net.biases = {double(b1)' , double(b2)', double(b3)'};
L = cell(50,1);
L(:) = {'poslin'};
small_net.layers{1} = L ;
L = cell(70,1);
L(:) = {'poslin'};
small_net.layers{2} = L ;
%%%%%%%%%%%%%%%%%%%%%%%%


dimp = 720*960;
dim2 = 720*960*12;

tic

res_trn = abs( Y - ( Directions * NN_eval(small_net , X)  + C ) );


threshold_normal = 10^(-15);
res_max = max( res_trn ,[] ,2 );
indices = find(res_max < threshold_normal);
res_max(indices,1) = threshold_normal;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
trn_time1 = toc;


clear X  Y  res_trn  


ell = rank;
Failure_chance_of_guarantee =  betacdf( guarantee , ell , Ns+1 -ell);


if Ns > 2000
    thelen = 2000;
else
    thelen = Ns;
end


if Ns > thelen
    chunck_size = thelen;
    Num_chuncks = floor(Ns / chunck_size);
    remainder = mod(Ns , chunck_size);
else
    chunck_size = Ns;
    Num_chuncks = 1;
    remainder = 0;
end
chunck_sizes = chunck_size * ones(1, Num_chuncks);
test_data_run = zeros(1,Num_chuncks);
res_test_time = zeros(1,Num_chuncks);
if remainder ~= 0
    chunck_sizes = [chunck_sizes , remainder]; %#ok<NASGU>
    test_data_run = zeros(1, Num_chuncks+1);
    res_test_time = zeros(1,Num_chuncks+1);
end



Rs = zeros(1, Ns);
ind = 0;

for nc=1:length(chunck_sizes)

    rng(nc)

    len = chunck_sizes(nc);
    Y_test_nc = zeros(720,960,12,len);
    X_test_nc = zeros(3*N_perturbed , len);

    cd(dir1)
    tic
    %%%%%%%%%%%%%%
    parfor i=1:len
        disp(['part ' num2str(nc) ' number ' num2str(i)])
        Rand = rand(3*N_perturbed,1);
        Rand_matrix = mat_generator_no_third(Rand , indices , [720,960,3]);
        d_at = zeros(720,960,3);
        for c=1:3
            d_at(:,:,c) = de * Rand_matrix(:,:,c) / std_vals(c) ;
        end
        Inp = Input + d_at;
        X_test_nc(:,i) = Rand;
        [Y_test_nc(:,:,:,i), ~] = BiSeNetFunc(Inp, params, 'InputDataPermutation', 'auto');
    end
    %%%%%%%%%%%%%

    test_data_run(nc) = toc;



    Y_test = reshape(Y_test_nc, [dim2 , len] );

    clear Y_test_nc


    cd(dir0)
    tic
    res_tst = abs(Y_test - ( Directions * NN_eval( small_net , X_test_nc) + C ));
    Rs(ind+1:ind+len) = max( res_tst ./ res_max);
    res_test_time(nc) = toc;

    clear Y_test X_test_nc res_tst

    ind = ind + len;

end



tic
Rs_sorted = sort(Rs);
R_star = Rs_sorted(:,ell);
Conf = R_star*res_max;

conformal_time = toc;

figure
plot(Conf)
 

dir_0 = fileparts(dir);
dir3 = [ dir_0  '/src'];
addpath(genpath(dir3))
dir_00 = fileparts(dir_0);
dir_000 = fileparts(dir_00);
dir4 = [ dir_000  '/nnv'];
addpath(genpath(dir4))

tic
H.V = [sparse(dim2,1) speye(dim2)];
H.C = sparse(1,dim2);
H.d = 0;
H.predicate_lb = -Conf;
H.predicate_ub =  Conf;
H.dim = dim2;
H.nVar= dim2;

I = Star();
I.V = [0.5*ones(3*N_perturbed,1)  eye(3*N_perturbed)];
I.C = zeros(1,3*N_perturbed);
I.d = 0;
I.predicate_lb = -0.5*ones(3*N_perturbed,1);
I.predicate_ub =  0.5*ones(3*N_perturbed,1);
I.dim =  3*N_perturbed;
I.nVar = 3*N_perturbed;



Principal_reach = ReLUNN_Reachability_starinit(I, small_net, 'approx-star');
Surrogate_reach = affineMap(Principal_reach , Directions , C);

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


Lb_pixels = reshape(Lb , [dimp, 12]);
Ub_pixels = reshape(Ub , [dimp, 12]);
%%%%%%%%%%%%%%
projection_time = toc;


clear  Surrogate_reach

classes = cell(720,960);

%%%  0 means: background   and    1 means: lung

for i=1:720
    for j = 1:960
        t = (j-1)*720+i;
        [L_star, my_class] = max(Lb_pixels(t,:));
        class_members = [];
        for k = 1:12
            if L_star <= Ub_pixels(t,k)
                class_members = [class_members , k];
            end
        end
        classes{i,j} = class_members;
    end
end




% Load the ONNX model
cd(dir1)
params = importONNXFunction('BiSeNet.onnx', 'BiSeNetFunc');


cd(dir2)
img = imread(image_name);  % <-- replace with a valid CamVid image filename
% img = imread('0006R0_f02190.png');  % <-- replace with a valid CamVid image filename

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

[BiSeNet, ~] = BiSeNetFunc(img_norm, params, 'InputDataPermutation', 'auto');


cd(dir0)

True_class = cell(720, 960);
for i =1:720
    for j=1:960
       [~, True_class{i,j}] = max(BiSeNet(i,j,:));
    end
end


robust = 0 ;
nonrobust = 0;
unknown = 0;
attacked = N_perturbed;
for i =1:720
    for j=1:960
        if (length(classes{i,j}) == 1) %#ok<ISCL>
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

dim_pic = 720*960;

RV = 100* robust / dim_pic;
disp(['Number of Robust pixels: ' num2str(robust)])
disp(['Number of non-Robust pixels: ' num2str(nonrobust)])
disp(['Number of unknown pixels: ' num2str(unknown)])
disp(['RV value: ' num2str(RV)])

disp([' Pr[    Pr[    RV_value is ' num2str(RV) '%  ]  >  '  num2str(guarantee) '  ]  >  '  num2str( 1- Failure_chance_of_guarantee)   ])

verification_runtime = train_data_run_1 + trn_time1 + sum(test_data_run) + sum(res_test_time) + ...
                       conformal_time + reachability_time + projection_time + Direction_Training_time + Model_training_time; %#ok<USENS>

disp([' The verification Runtime is: ' num2str(verification_runtime/60) ' minutes.'   ])

name = ['CI_result_middle_guarantee_ReLU_relaxed_eps_' num2str(delta_rgb) '_Npertubed_' num2str(N_perturbed) '.mat'];

save(name, "robust", "nonrobust", "attacked", "unknown", "True_class", "classes", "Conf"        , ...
                                                              "N", "N_dir", "de", "ell", "Lb_pixels", "Ub_pixels", "Ns", "R_star"         , ...
                                                              "res_max", "RV", "threshold_normal" , "verification_runtime", ...
                                                              "train_data_run_1", "trn_time1", "test_data_run", "res_test_time", ...
                       "conformal_time" , "reachability_time", "projection_time", "Direction_Training_time", "Model_training_time");




end