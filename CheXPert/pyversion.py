import os
import sys
import torch
import numpy as np
import cv2
from tqdm import tqdm
from scipy.stats import beta
from time import time
from torch.cuda.amp import autocast
import matlab.engine
import scipy.io
import onnxruntime as ort


# --- Configuration ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


N_perturbed = 50  # adjust as needed
image_name = 'CHNCXR_0005_0.png'
start_loc = (512-20, 512-20)  # modify as needed
Ns = 10
Nsp = 2
rank = 9
guarantee = 0.99
delta_rgb = 25
de = delta_rgb / 255.0
Nt = 4  # Total number of samples
N_dir = 2  # Number of samples used for direction learning
epsilon = 0.001

def UnetFunc(x, batch_size=2):
    x = x.to(torch.float16)  # Use half precision
    x_numpy = x.cpu().numpy().astype(np.float32)
    results = []
    for i in range(0, x_numpy.shape[0], batch_size):
        batch = x_numpy[i:i+batch_size]
        #with autocast():  # Automatically use mixed precision
        with torch.amp.autocast('cuda'):
            output = ort_session.run(None, {'input': batch})
        results.append(torch.tensor(output[0]).to(device))
    return torch.cat(results, dim=0)


# --- Paths ---
base_dir   = os.getcwd()
model_path = os.path.join(base_dir, 'lung_segmentation.onnx')
image_path = os.path.join(base_dir, 'images',  image_name)

ort_session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])



img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # ensures it's grayscale
img = cv2.resize(img, (512, 512))
img = img.astype(np.float32) / 255.0
img = img.reshape(1, 1, 512, 512)
at_im = img.copy()

# --- Apply darkening attack ---
ct = 0
indices = []

_, _, H, W = img.shape
for i in range(start_loc[0], H):
    for j in range(start_loc[1], W):
        if np.min(img[:,:,i, j]) > 150 / 255.0:
            at_im[:,:,i, j] = 0.0
            indices.append([i, j])
            ct += 1
            if ct == N_perturbed:
                print(f"{N_perturbed} pixels found.")
                break
    if ct == N_perturbed:
        break

indices = np.array(indices)

# --- Run the model ---
at_im_tensor = torch.from_numpy(at_im).to(device)
# with torch.no_grad():
#     output = UnetFunc(at_im_tensor)


# import matplotlib.pyplot as plt

# # Convert output to binary: 1 if > 0, else 0
# binary_output = (output > 0).int().squeeze().detach().cpu().numpy()

# # Plotting as black and white
# plt.figure(figsize=(6, 6))
# plt.imshow(binary_output, cmap='gray', vmin=0, vmax=1)
# plt.title('Binarized Segmentation Output')
# plt.axis('off')
# plt.show()


def mat_generator_no_third(values: torch.Tensor, indices: torch.Tensor, original_dim: tuple):
    Matrix = torch.zeros(original_dim, device=values.device, dtype=values.dtype)
    N_perturbed = indices.shape[0]
    t = 0
    for i in range(N_perturbed):
        row, col = indices[i]
        Matrix[:,:,row, col] = values[:,t].unsqueeze(1)
        t += 1
    return Matrix


Input = at_im_tensor


torch.manual_seed(0)


def generate_data_chunk(N, indices, de, Inputs, model):
    
    """ Function to generate the training data for one instance in parallel. """
    Rand = torch.rand(N, 3 * N_perturbed).to(device)
    Rand_matrix = mat_generator_no_third(Rand, indices, (N, 1, 512, 512))
    d_at = de * Rand_matrix
    Inp = Inputs + d_at
    Inp_tensor = Inp.float()

    with torch.no_grad():
        out = model(Inp_tensor)
    
    return out, Rand



t0 = time()



Inputs = Input.repeat(Nt,1,1,1)
Y, X = generate_data_chunk(Nt, indices, de, Inputs, UnetFunc)


train_data_run_1 = time() - t0


Y = Y.view(Y.shape[0], -1)


assert N_dir <= Nt, "Requested more samples than available!"

selected_indices = torch.randperm(Nt)[:N_dir]

X_dir = X[selected_indices, ...].to(device)
Y_dir = Y[selected_indices, ...].to(device)


from Direction_training import compute_directions

Directions, Direction_Training_time = compute_directions(Y_dir, device, 5)

Directions = torch.stack([d.squeeze(-1) for d in Directions])


C = 20 * (epsilon * Y.mean(dim=0) + (0.05 - epsilon) * 0.5 * (Y.min(dim=0).values + Y.max(dim=0).values))
dY = Y - C.unsqueeze(0)
dYV = dY @ Directions.T

torch.cuda.empty_cache()


from Training_ReLU import Trainer_ReLU
current_dir = os.getcwd()
save_path = os.path.join(current_dir, 'trained_relu_weights_2h_norm.mat')
epochs = 50
small_net, Model_training_time = Trainer_ReLU(X, dYV, device, epochs, save_path)

with torch.no_grad():
    pred = small_net(X) 

    approx_Y = pred @ Directions  + C.unsqueeze(0)  # shape: same as Y

t0 = time()
residuals = (Y - approx_Y).abs()
threshold_normal = 1e-5
res_max = residuals.max(dim=0).values
res_max[res_max < threshold_normal] = threshold_normal

trn_time1 = time()-t0


ell = rank
Failure_chance_of_guarantee = beta.cdf(guarantee, ell, Ns + 1 - ell)


thelen = min(Ns, Nsp)
if Ns > thelen:
    chunck_size = thelen
    Num_chunks = Ns // chunck_size
    remainder = Ns % chunck_size
else:
    chunck_size = Ns
    Num_chunks = 1
    remainder = 0

chunk_sizes = [chunck_size] * Num_chunks
if remainder != 0:
    chunk_sizes.append(remainder)

Rs = torch.zeros(Ns, requires_grad=False)
ind = 0
test_data_run = []
res_test_time = []



for nc, curr_len in enumerate(chunk_sizes):
    torch.manual_seed(nc+1)

    t0 = time()
    
    Inputs = Input.repeat(curr_len,1,1,1)
    
    Y_test_nc, X_test_nc = generate_data_chunk(curr_len, indices, de, Inputs, UnetFunc)


    test_data_run.append(time() - t0)

    Y_test = Y_test_nc.view(Y_test_nc.shape[0], -1)
    del Y_test_nc

    t1 = time()
    pred = small_net(X_test_nc) @ Directions  + C.unsqueeze(0)  # shape: (dim2, curr_len)
    res_tst = (Y_test - pred).abs()
    Rs[ind:ind + curr_len] = torch.max(res_tst / res_max.unsqueeze(0), dim=1).values
    res_test_time.append(time() - t1)

    del Y_test,  X_test_nc, res_tst
    ind += curr_len


t0 = time()

with torch.no_grad():
    Rs_sorted = torch.sort(Rs).values
    R_star = Rs_sorted[ell]  # Assuming `ell` is defined
    Conf = R_star * res_max

conformal_time = time() - t0

current_dir = os.getcwd()
save_path = os.path.join(current_dir, 'python_data.mat')
c = C.cpu().numpy()
conf = Conf.cpu().numpy()
directions = Directions.cpu().numpy()

scipy.io.savemat(save_path, {
    'Conf': conf, 'C': c, 'Directions': directions})

del conf, c, directions


eng = matlab.engine.start_matlab()


matlab_code = r"""


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

"""

eng.eval(matlab_code, nargout=0)

eng.quit()


current_dir = os.getcwd()
mat_file_path = os.path.join(current_dir, 'Matlab_data.mat')
mat_data = scipy.io.loadmat(mat_file_path)

Lb_pixels = torch.tensor(mat_data['Lb_pixels'], dtype=torch.float32)
Ub_pixels = torch.tensor(mat_data['Ub_pixels'], dtype=torch.float32)
projection_time = float(mat_data['projection_time'].item())
reachability_time = float(mat_data['reachability_time'].item())
conformal_time2 = float(mat_data['conformal_time2'].item())


start_time = time()


classes = [[None for _ in range(512)] for _ in range(512)]

for i in range(512):
    for j in range(512):
        t = i * 512 + j  
        lb = Lb_pixels[t].item()
        ub = Ub_pixels[t].item()

        if lb > 0:
            class_members = [1]
        elif ub <= 0:
            class_members = [0]
        else:
            class_members = [0, 1]
        classes[i][j] = class_members


img_tensor = torch.from_numpy(img).to(device)
output = UnetFunc(img_tensor)

output_np = output.squeeze().cpu().numpy()  # shape: [512, 512]

True_class = [[int(output_np[i, j] > 0) for j in range(512)] for i in range(512)]



# Initialize counters
robust = 0
nonrobust = 0
unknown = 0
attacked = N_perturbed  # Assuming this is defined

for i in range(512):
    for j in range(512):
        if len(classes[i][j]) == 1:
            if classes[i][j] == [True_class[i][j]]:
                robust += 1
            else:
                nonrobust += 1
        else:
            if True_class[i][j] in classes[i][j]:
                unknown += 1
            else:
                nonrobust += 1

# Compute the robustness percentage
dim_pic = 512*512
RV = 100 * robust / dim_pic

print(f"Number of Robust pixels: {robust}")
print(f"Number of non-Robust pixels: {nonrobust}")
print(f"Number of unknown pixels: {unknown}")
print(f"RV value: {RV}")

print(f"Pr[RV value > {guarantee}%] > {1 - Failure_chance_of_guarantee}")


verification_runtime = train_data_run_1 + trn_time1 + sum(test_data_run) + sum(res_test_time) + \
                       conformal_time + reachability_time + projection_time + Direction_Training_time + Model_training_time

print(f"The verification runtime is: {verification_runtime / 60:.2f} minutes.")


save_dict = {
    "robust": robust,
    "nonrobust": nonrobust,
    "attacked": attacked,
    "unknown": unknown,
    "True_class": True_class,
    "classes": classes,
    "Conf": Conf,
    "Nt": Nt,
    "N_dir": N_dir,
    "de": de,
    "ell": ell,
    "Lb_pixels": Lb_pixels,
    "Ub_pixels": Ub_pixels,
    "Ns": Ns,
    "R_star": R_star,
    "res_max": res_max,
    "RV": RV,
    "verification_runtime": verification_runtime,
    "threshold_normal": threshold_normal,
    "train_data_run_1": train_data_run_1,
    "trn_time1": trn_time1,
    "test_data_run": test_data_run,
    "res_test_time": res_test_time,
    "conformal_time": conformal_time,
    "reachability_time": reachability_time,
    "projection_time": projection_time,
    "Direction_Training_time": Direction_Training_time,
    "Model_training_time": Model_training_time
}


for key, val in save_dict.items():
    if isinstance(val, torch.Tensor):
        save_dict[key] = val.cpu()
    elif isinstance(val, list):
        save_dict[key] = [v.cpu() if isinstance(v, torch.Tensor) else v for v in val]
    elif isinstance(val, dict):
        save_dict[key] = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in val.items()}
        
save_name = f"CI_result_middle_guarantee_ReLU_relaxed_eps_{delta_rgb}_Npertubed_{N_perturbed}.pt"
    
torch.save(save_dict, save_name)