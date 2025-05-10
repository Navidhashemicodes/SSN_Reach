import os
import sys
import torch
import torchvision.transforms as transforms
from torchvision.io import read_image
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.stats import beta
from time import time
from joblib import Parallel, delayed
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matlab.engine
import scipy.io


# --- Configuration ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_perturbed = 50  # adjust as needed
image_name = '0006R0_f02190.png'
start_loc = (360-20, 480-20)  # modify as needed
Ns = 20
rank = 19
guarantee = 0.99
# --- Paths ---
base_dir = os.path.dirname(os.getcwd())
model_path = os.path.join(base_dir, 'Pytorch_repo', 'BiSeNet-master', 'model', 'best_dice_loss_miou_0_655.pth')
class_path = os.path.join(base_dir, 'Pytorch_repo', 'BiSeNet-master', 'model')
image_path = os.path.join(base_dir, 'Pytorch_repo', 'CamVid', 'test', image_name)

# --- Load the model ---
sys.path.append(class_path)

from build_BiSeNet import BiSeNet

num_classes = 12  # Adjust depending on the dataset
model = BiSeNet(num_classes, 'resnet18')
# model.load_state_dict(torch.load(model_path, map_location=device))
model.load_state_dict(torch.load(model_path))
# model.to(device)
model.eval()

# --- Load and preprocess image ---
img = Image.open(image_path).convert('RGB')
img_np = np.array(img).astype(np.float32) / 255.0  # shape: (H, W, 3), range [0, 1]
at_im = img_np.copy()

# --- Apply darkening attack ---
ct = 0
indices = []

H, W, _ = img_np.shape
for i in range(start_loc[0], H):
    for j in range(start_loc[1], W):
        if np.min(img_np[i, j, :]) > 150 / 255.0:
            at_im[i, j, :] = 0.0
            indices.append([i, j])
            ct += 1
            if ct == N_perturbed:
                print(f"{N_perturbed} pixels found.")
                break
    if ct == N_perturbed:
        break

indices = np.array(indices)

# --- Normalize image ---
mean_vals = np.array([0.485, 0.456, 0.406])
std_vals = np.array([0.229, 0.224, 0.225])

at_im_norm = (at_im - mean_vals) / std_vals

# Transpose and convert to tensor
at_im_tensor =torch.tensor(at_im_norm.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0)
# --- Run the model ---
with torch.no_grad():
    output = model(at_im_tensor)

# --- Output ---
predicted_classes = output.argmax(dim=1).squeeze().cpu().numpy()
print("Inference complete. Predicted shape:", predicted_classes.shape)

# Note: Call MATLAB only for Star-based reachability using saved H, I sets externally


# Constants and config
N = 10  # Total number of samples
N_dir = 5  # Number of samples used for direction learning
img_shape = (720, 960, 3)
delta_rgb = 25
std_vals = torch.tensor([0.229, 0.224, 0.225])

def mat_generator_no_third(values: torch.Tensor, indices: torch.Tensor, original_dim: tuple):
    Matrix = torch.zeros(original_dim, device=values.device, dtype=values.dtype)
    N_perturbed = indices.shape[0]
    t = 0
    for c in range(3):
        for i in range(N_perturbed):
            row, col = indices[i]
            Matrix[row, col, c] = values[t]
            t += 1
    return Matrix


Input = at_im_tensor

# Example perturbed pixel locations (randomized for example)
torch.manual_seed(0)
indices = torch.randint(0, 720, (N_perturbed, 2))

# Data placeholders
Y = torch.zeros(720, 960, 12, N, dtype=torch.float32)
X = torch.zeros(3 * N_perturbed, N, dtype=torch.float32)

de = delta_rgb / 255.0


torch.manual_seed(0)


def generate_data_chunk(i, indices, de, std_vals, Input, model):
    
    """ Function to generate the training data for one instance in parallel. """
    Rand = torch.rand(3 * N_perturbed)
    Rand_matrix = mat_generator_no_third(Rand, indices, (720, 960, 3))
    d_at = torch.zeros((720, 960, 3))
    for c in range(3):
        d_at[:, :, c] = de * Rand_matrix[:, :, c] / std_vals[c]
    d_at_tensor = d_at.permute(2, 0, 1).unsqueeze(0) 
    Inp = Input + d_at_tensor
    # Inp_tensor = Inp.permute(2, 0, 1).unsqueeze(0).float()
    Inp_tensor = Inp.float()

    # Direct BiSeNet inference (model is already preloaded)
    with torch.no_grad():
        out = model(Inp_tensor)  # (1, 3, 720, 960) -> (1, 12, 720, 960)
    out = out.squeeze(0).permute(1, 2, 0)  # (720, 960, 12)
    
    return out, Rand

# Initialize Y and X
Y = torch.zeros((720, 960, 12, N))
X = torch.zeros((3 * N_perturbed, N))

# Start timing
t0 = time()

# for i in range(0,N):
#     FF = generate_train_data_chunk(i, indices, de, std_vals, Input, model)

# Start the pool
results = Parallel(n_jobs=10, backend='loky', verbose = 10)(
    delayed(generate_data_chunk)(i, indices, de, std_vals, Input, model) for i in range(N))



# Collect results and assign them
for i, (out, Rand) in enumerate(results):
    Y[:, :, :, i] = out
    X[:, i] = Rand

train_data_run_1 = time() - t0


Y = Y.view(-1, X.shape[-1])


assert N_dir <= N, "Requested more samples than available!"

# Randomly select sample indices
selected_indices = torch.randperm(N)[:N_dir]

# Select and move to GPU
X_dir = X[:, selected_indices].to(device)  # (3*N_perturbed, N_dir)
Y_dir = Y[:, selected_indices].to(device)  # (720, 960, 19, N_dir)


from Direction_training import compute_directions

Directions, Direction_Training_time = compute_directions(Y_dir, device)

Directions = torch.stack([d.squeeze(-1) for d in Directions]).cpu()


# Compute stats of Y on CPU
min_Y, _ = Y.min(dim=1, keepdim=True)   # (720, 1, 19, N_total)
max_Y, _ = Y.max(dim=1, keepdim=True)
mean_Y = Y.mean(dim=1, keepdim=True)

# Compute C on CPU, then move to GPU
C = 20.0 * (0.01 * mean_Y + (0.05 - 0.01) * 0.5 * (min_Y + max_Y))  # (720, 1, 19, N_total)

# Reshape Y and C to (dim, N_total) for matmul

# Compute dYV = Directionsᵀ @ (Y - C)
dYV = Directions @ (Y - C)  # (N_dir, N_total)

# Free GPU memory by deleting Directions_gpu
torch.cuda.empty_cache()



from Training_ReLU import Trainer_ReLU
current_dir = os.getcwd()
save_path = os.path.join(current_dir, 'trained_relu_weights_2h_norm.mat')
epochs = 50
small_net, Model_training_time = Trainer_ReLU(X, dYV, device, epochs, save_path)
small_net = small_net.cpu()

# Step 4: Evaluate network
with torch.no_grad():
    pred = small_net(X.T).T  

    # Directions @ pred + C
    approx_Y = Directions.T @ pred + C  # shape: same as Y

# Step 5: Compute residuals
t0 = time()
residuals = (Y - approx_Y).abs()
threshold_normal = 1e-15
res_max = residuals.max(dim=1).values
res_max[res_max < threshold_normal] = threshold_normal

trn_time1 = time()-t0


# Assumptions: Input, std_vals, indices, de, Directions, C, res_max, small_net, model, and params are preloaded

# Parameters
ell = rank
Failure_chance_of_guarantee = beta.cdf(guarantee, ell, Ns + 1 - ell)

thelen = min(Ns, 2000)
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

    Y_test_nc = torch.zeros((720, 960, 12, curr_len))
    X_test_nc = torch.zeros((3 * N_perturbed, curr_len))

    t0 = time()

    # Parallelize the inner loop using multiprocessing Pool

    results = Parallel(n_jobs=10, backend='loky', verbose = 10)(
        delayed(generate_data_chunk)(i, indices, de, std_vals, Input, model) for i in range(Ns))

    # Collect results and assign them
    for i, (out, Rand) in enumerate(results):
        Y_test_nc[:, :, :, i] = out
        X_test_nc[:, i] = Rand

    test_data_run.append(time() - t0)

    Y_test = Y_test_nc.reshape((-1, curr_len))
    del Y_test_nc

    t1 = time()
    pred = Directions.T @ small_net(X_test_nc.T).T + C  # shape: (dim2, curr_len)
    res_tst = (Y_test - pred).abs()
    Rs[ind:ind + curr_len] = torch.max(res_tst / res_max[:, None], dim=0).values
    res_test_time.append(time() - t1)

    del Y_test, X_test_nc, res_tst
    ind += curr_len


# Assuming Rs and res_max are available from the previous steps
# Conf computation
t0 = time()

with torch.no_grad():
    Rs_sorted = torch.sort(Rs).values
    R_star = Rs_sorted[ell]  # Assuming `ell` is defined
    Conf = R_star * res_max

conformal_time = time() - t0

current_dir = os.getcwd()
save_path = os.path.join(current_dir, 'python_data.mat')
c = C.numpy()
conf = Conf.numpy()
directions = Directions.numpy()

scipy.io.savemat(save_path, {
    'Conf': conf, 'C': c, 'Directions': directions})

del conf, c, directions


# Start MATLAB engine
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

dim2 = 720*960*12
dimp = 720*960

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
Surrogate_reach = affineMap(Principal_reach , Directions.' , C);

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



Lb_max, _ = torch.max(Lb_pixels, dim=1, keepdim=True)  # Shape: [720*960, 1]

# Compare with upper bounds
mask = Lb_max <= Ub_pixels  # Shape: [720*960, 12], boolean

# Convert to CPU and Python list for interpretation
mask_np = mask.cpu().numpy()

# Now create the per-pixel class_members list
classes = [[[] for _ in range(960)] for _ in range(720)]

for t in range(720 * 960):
    i = t % 720
    j = t // 720
    class_members = [k for k in range(12) if mask_np[t, k]]
    classes[i][j] = class_members



img = Image.open(image_path).convert('RGB')
img_np = np.array(img).astype(np.float32) / 255.0  # shape: (H, W, 3), range [0, 1]

img_norm = (img_np - mean_vals) / std_vals

img_tensor = img_norm.permute(2, 0, 1).unsqueeze(0).float()
output = model(img_tensor)  # Assuming the model is already loaded

# Get the class predictions from the output
True_class = {}
for i in range(720):
    for j in range(960):
        _, True_class[(i, j)] = torch.max(output[0, :, i, j], 0)  # Assuming output shape is [1, C, H, W]

# Initialize counters
robust = 0
nonrobust = 0
unknown = 0
attacked = N_perturbed  # Assuming this is defined

for i in range(720):
    for j in range(960):
        if len(classes[i][j]) == 1:
            if classes[i][j] == True_class[(i, j)]:
                robust += 1
            else:
                nonrobust += 1
        else:
            if True_class[(i, j)] in classes[i][j]:
                unknown += 1
            else:
                nonrobust += 1

# Compute the robustness percentage
dim_pic = 720 * 960
RV = 100 * robust / dim_pic

print(f"Number of Robust pixels: {robust}")
print(f"Number of non-Robust pixels: {nonrobust}")
print(f"Number of unknown pixels: {unknown}")
print(f"RV value: {RV}")

# Assuming `guarantee` and `Failure_chance_of_guarantee` are defined
print(f"Pr[RV value > {guarantee}%] > {1 - Failure_chance_of_guarantee}")

# Calculate the total runtime


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
    "N": N,
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

