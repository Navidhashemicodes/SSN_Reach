function verify_M2NIST_on_CNNS( start_loc, N_perturbed, image_number, model_name, model_number, delta_rgb, N, N_dir, Ns , rank, guarantee  )

% %%% Nice setting
% Ns = 100,000
% start_loc = [64-10 , 84-10]
% rank = 99,999
% guarantee = 0.9999  -->  confidence = 0.9995
% N_perturbed = 17, 34, 51, 68, 85, 102
% image_number = 1, ..., 30
% delta_rgb = 3
% N = 2,000
% N_dir = 2,000
%model_name = m2nist_dilated_72iou_24layer.mat
%%%%%%%%%%%%%%%%%%%

dim_1 = 64;
dim_2 = 84;
n_class = 11;
de = delta_rgb;

dir0 = pwd;
dir = fileparts(dir0);

dir1 = [ dir  '/models'];
dir2 = [ dir  '/Test_images'];
% Load the ONNX model
cd(dir1)
load(model_name)
layers = net.Layers;  
newLayers = layers(1:end-2);
Net = SeriesNetwork(newLayers);
cd(dir0)

cd(dir2)
load("m2nist_6484_test_images.mat")
img = im_data(:,:,image_number);

ct = 0;
flag = 0;
at_im = img;
indices = zeros(N_perturbed , 2);
for i=start_loc(1):dim_1
    for j=start_loc(2):dim_2
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

Input = at_im;

cd(dir0)
rng(0)

Y = zeros(dim_1,dim_2,n_class,N);
X = zeros(N_perturbed , N);
tic
%%%%%%%%%%%%%%
parfor i=1:N
    disp(i)
    Rand = rand(N_perturbed,1);
    Rand_matrix = mat_generator(Rand , indices , [dim_1,dim_2]);
    d_at = de * Rand_matrix ;
    Inp = single(Input + d_at);
    X(:,i) = Rand;
    Y(:,:,:,i)= Net.predict(Inp);
end
%%%%%%%%%%%%%
train_data_run_1 = toc;


if N_dir >2000
    leng = 2000;
else
    leng = N_dir;
end
ind = 0;
if mod(N_dir , leng) == 0
    last_i = N_dir/leng ;
else
    error('Considering M2NIST_dataset, if N_dir is more than 2000 it should be devisible by 2000.')
end
for i=1:last_i
    eval(['Y' num2str(i) ' = Y(:,:,:,ind+1:ind+leng);' ]);
    ind = ind+leng;
end

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

mat_file_path = [dir0 '/Direction_data.mat'];

command = sprintf(['python Direction_trainer.py --mat_file_path "%s" ' ...
                   '--num_files %d --N_dir %d '], ...
                   mat_file_path,  last_i, N_dir);

status = system(command);

n1 = numel(Y(:,:,:,1));
Y = reshape(Y, [n1 , N]);

C = 20* (  0.01*(mean(Y,2))   +  (0.05-0.01) * 0.5*( min(Y , [], 2) + max(Y , [] , 2) ));

load('directions.mat')

% pyenv
% npz = py.numpy.load('directions.npz');
% shape = cellfun(@double, cell(npz{'Directions'}.shape));
% Directions_py = py.numpy.array(npz{'Directions'});
% Directions = single(Directions_py.reshape(int64(shape)));  % use reshape to preserve dimensions
% Direction_Training_time = double(npz{'Direction_Training_time'});
% clear Directions_py  npz


dYV  = Directions' * (Y - C );

save("Reduced_dimension.mat", "dYV", "X");


system('python Trainer_ReLU.py')


load("trained_relu_weights_2h_norm.mat")
small_net.weights = {double(W1) , double(W2), double(W3)};
small_net.biases = {double(b1)' , double(b2)', double(b3)'};
L = cell(size(W1,1),1);
L(:) = {'poslin'};
small_net.layers{1} = L ;
L = cell(size(W2,1),1);
L(:) = {'poslin'};
small_net.layers{2} = L ;
%%%%%%%%%%%%%%%%%%%%%%%%


dimp = dim_1*dim_2;
dim2 = dim_1*dim_2*n_class;

tic

res_trn = abs( Y - ( Directions * NN_eval(small_net , X)  + C ) );


threshold_normal = 10^(-5);
res_max = max( res_trn ,[] ,2 );
theindices = find(res_max < threshold_normal);
res_max(theindices,1) = threshold_normal;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
trn_time1 = toc;


clear X  Y  res_trn  


ell = rank;
Failure_chance_of_guarantee =  betacdf( guarantee , ell , Ns+1 -ell);


if Ns > 60000
    thelen = 60000;
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
    Y_test_nc = zeros(dim_1,dim_2,n_class,len);
    X_test_nc = zeros(N_perturbed , len);

    
    tic
    %%%%%%%%%%%%%%
    parfor i=1:len
        disp(['part ' num2str(nc) ' number ' num2str(i)])
        Rand = rand(N_perturbed,1);
        Rand_matrix = mat_generator(Rand , indices , [dim_1,dim_2]);
        d_at = de * Rand_matrix;
        Inp = Input + d_at;
        X_test_nc(:,i) = Rand;
        Y_test_nc(:,:,:,i) = Net.predict(Inp);
    end
    %%%%%%%%%%%%%

    test_data_run(nc) = toc;



    Y_test = reshape(Y_test_nc, [dim2 , len] );

    clear Y_test_nc

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
I.V = [0.5*ones(N_perturbed,1)  eye(N_perturbed)];
I.C = zeros(1,N_perturbed);
I.d = 0;
I.predicate_lb = -0.5*ones(N_perturbed,1);
I.predicate_ub =  0.5*ones(N_perturbed,1);
I.dim =  N_perturbed;
I.nVar = N_perturbed;



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


Lb_pixels = reshape(Lb , [dimp, n_class]);
Ub_pixels = reshape(Ub , [dimp, n_class]);
%%%%%%%%%%%%%%
projection_time = toc;


clear  Surrogate_reach

classes = cell(dim_1,dim_2);

%%%  0 means: background   and    1 means: lung

for i=1:dim_1
    for j = 1:dim_2
        t = (j-1)*dim_1+i;
        [L_star, my_class] = max(Lb_pixels(t,:));
        class_members = [];
        for k = 1:n_class
            if L_star <= Ub_pixels(t,k)
                class_members = [class_members , k];
            end
        end
        classes{i,j} = class_members;
    end
end




SSN = net.predict(Input);

True_class = cell(dim_1, dim_2);
for i =1:dim_1
    for j=1:dim_2
       [~, True_class{i,j}] = max(SSN(i,j,:));
    end
end


robust = 0 ;
nonrobust = 0;
unknown = 0;
attacked = N_perturbed;
for i =1:dim_1
    for j=1:dim_2
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

dim_pic = dim_1*dim_2;

RV = 100* robust / dim_pic;
disp(['Number of Robust pixels: ' num2str(robust)])
disp(['Number of non-Robust pixels: ' num2str(nonrobust)])
disp(['Number of unknown pixels: ' num2str(unknown)])
disp(['RV value: ' num2str(RV)])

disp([' Pr[    Pr[    RV_value is ' num2str(RV) '%  ]  >  '  num2str(guarantee) '  ]  >  '  num2str( 1- Failure_chance_of_guarantee)   ])

verification_runtime = train_data_run_1 + trn_time1 + sum(test_data_run) + sum(res_test_time) + ...
                       conformal_time + reachability_time + projection_time + Direction_Training_time + Model_training_time; %#ok<USENS>

disp([' The verification Runtime is: ' num2str(verification_runtime/60) ' minutes.'   ])

name = ['CI_result_middle_guarantee_ReLU_relaxed_eps_' num2str(delta_rgb) '_image_number_' num2str(image_number) '_N_perturbed_' num2str(N_perturbed) '_model_number' num2str(model_number) '.mat'];

save(name, "robust", "nonrobust", "attacked", "unknown", "True_class", "classes", "Conf"        , ...
                                                              "N", "N_dir", "de", "ell", "Lb_pixels", "Ub_pixels", "Ns", "R_star"         , ...
                                                              "res_max", "RV", "threshold_normal" , "verification_runtime", ...
                                                              "train_data_run_1", "trn_time1", "test_data_run", "res_test_time", ...
                       "conformal_time" , "reachability_time", "projection_time", "Direction_Training_time", "Model_training_time");




end