import os
import torch
import numpy as np
from scipy.stats import beta
from time import time
from torch.cuda.amp import autocast
import matlab.engine
import scipy.io

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random
from time import time
import gc

import os
import sys
import numpy as np
from PIL import Image
# from tqdm import tqdm
from scipy.stats import beta
from time import time
from joblib import Parallel, delayed
# import matlab.engine
import scipy.io
import matlab.engine

# loading models
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation # SegFormer
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation # Mask2Former

# loading images
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# for inference
import torch.nn.functional as F

# for processing args
import yaml
import argparse


torch.manual_seed(0)

def load_config(config_path="config.yaml"):
    """Load config file"""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    return parser.parse_args()

def batched_inference(inputs, model, batch_size, device='cuda'):
    model.eval()
    results = []
    with torch.no_grad():
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i+batch_size].to(device)
            
            if model_name == "hrnet":
                output = model(batch)  # [B, C, H, W]
                # pred = output.argmax(1).squeeze(0).cpu().numpy()  # [H, W]
                # pred = F.interpolate(output, size=(512, 1024), mode='bilinear', align_corners=True)
                predicted_classes = F.interpolate(output, size=(1024, 2048), mode='bilinear', align_corners=True)
                predicted_classes = torch.argmax(predicted_classes, dim=1).squeeze(0)  # shape: (512, 1024)

            elif model_name == "segformer":
                outputs = model(batch)  # This will now be [1, 3, 1024, 1024]
                logits = outputs.logits  # shape: [1, 19, 128, 128]

                # Upsample to 1024x1024
                outputs = F.interpolate(logits, size=(1024, 1024), mode='bilinear', align_corners=False)
                
            else: # model_name == "mask2former"
                outputs = model(batch) # this might break...
                predicted_classes = processor.post_process_semantic_segmentation(outputs, target_sizes=[(1024, 1024)])[0]
                predicted_classes = predicted_classes.cpu().numpy()
                # gt = label.squeeze(0).numpy()  # ground truth (H, W)

            results.append(outputs.cpu())

    return torch.cat(results, dim=0)


args = parse_args()
config = load_config(args.config)


# if len(sys.argv) != 12:
#     raise ValueError(f"Incorrect number of arguments {len(sys.argv)}... should be {12}.")

image_name = config['image_name']
image_parent_dir = config['image_parent_dir']
label_name = config['label_name']
height = config['height']
width = config['width']
model_name = config['model_name']
num_classes = config['num_classes']
N_perturbed = config['N_perturbed'] # number of pixels perturbed (if it's RGB then N_perturbed * 3)
Ns = config['Ns'] # size of calibration dataset
rank = config['rank'] # we choose a sample at specific rank? make this a command-line argument... something like Ns-1 or 1900 if Ns=2000
delta_rgb = config['delta_rgb'] # amount of pixel perturbation allowed, e.g. 25 if wanted 25/255 perturbation
Nt = config['N'] # total number of samples for learning the principal directions (train the surrogate model)
# we'll want to increase N for the experiments... something like 2000?
N_dir = config['N_dir'] # number of samples used for direction learning
# we'll also want to increase N_dir... something like 150


# --- Configuration ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
start_loc = ((height // 2)-20, (width // 2)-20) # this is for CamVid (720x960) ... start from middle for the darkening attack; pixels with value > 150 will be darkened (set to 0)
# the attacked image is the one that we will perturb to create infinite set
guarantee = 0.99
de = delta_rgb / 255.0

# --- Paths ---
base_dir = os.path.dirname(os.getcwd()) # /path/to/SSN_Reach
base_dir = os.path.join(base_dir, 'CityScapesExperiments') # /path/to/SSN_Reach/CityScapesExperiments
image_path = os.path.join(base_dir, 'data', 'leftImg8bit_trainvaltest', 'val', image_parent_dir, image_name)
model_path = os.path.join(base_dir, 'models', 'hrnet_cs_8090_torch11.pth')
label_path = os.path.join(base_dir, 'data', 'gtFine_trainvaltest', 'val', image_parent_dir, label_name)



start_loc = (512-20, 512-20)  # modify as needed
Nsp = 2
rank = 9
guarantee = 0.99
de = delta_rgb / 255.0
# Nt = 4  # Total number of samples
# N_dir = 2  # Number of samples used for direction learning
epsilon = 0.001

# --- Load the model ---
if model_name == "hrnet":
    # Build and load model
    model = get_seg_model(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hrnet_state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(hrnet_state_dict, strict=False)

elif model_name == "segformer":
    processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b2-finetuned-cityscapes-1024-1024")
    processor.size = {"height": 1024, "width": 1024}
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b2-finetuned-cityscapes-1024-1024").to(device)
    # feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b2-finetuned-cityscapes-1024-1024")

else: # model_name == "mask2former"
    processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-cityscapes-semantic")
    model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-cityscapes-semantic").to(device)

model.eval()

# --- Load and preprocess image ---
# TODO: Fix this section...
ID_TO_TRAINID = {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4,
    17: 5, 19: 6, 20: 7, 21: 8, 22: 9,
    23: 10, 24: 11, 25: 12, 26: 13,
    27: 14, 28: 15, 31: 16, 32: 17, 33: 18
}

def convert_to_train_ids(label):
    label_copy = 255 * torch.ones_like(label)  # 255 = ignore
    for k, v in ID_TO_TRAINID.items():
        label_copy[label == k] = v
    return label_copy

if model_name == "hrnet":
    img_transform = T.Compose([
        T.Resize((1024, 2048)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    label_transform = T.Compose([
        T.Resize((1024, 2048), interpolation=Image.NEAREST),
        T.Lambda(lambda img: torch.from_numpy(np.array(img)).long())
    ])

elif model_name == "segformer":
    img_transform = T.Compose([
        T.Resize((1024, 1024)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    label_transform = T.Compose([
        T.Resize((1024, 1024), interpolation=Image.NEAREST),
        T.Lambda(lambda img: torch.from_numpy(np.array(img)).long())
    ])
    
else: # model_name == "mask2former"
    pass


resize_transform = T.Resize((1024, 1024))
label_transform = T.Lambda(lambda img: torch.from_numpy(np.array(img)).long())

img = Image.open(image_path).convert('RGB')
# img = resize_transform(img) # resized to 1024x1024
img_np = np.array(img).astype(np.float32) / 255.0  # shape: (H, W, 3), range [0, 1]
at_im = img_np.copy()

label = Image.open(label_path)
label = resize_transform(label)
label = label_transform(label)

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
# TODO:
mean_vals = np.array([0.485, 0.456, 0.406])
std_vals = np.array([0.229, 0.224, 0.225])

# at_im_norm = (at_im - mean_vals) / std_vals
# Transpose and convert to tensor
# at_im_tensor = torch.tensor(at_im_norm.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0)

at_im_uint8 = (at_im * 255).astype(np.uint8)
at_pil = Image.fromarray(at_im_uint8)
at_im_tensor = processor(images=at_pil, return_tensors="pt")["pixel_values"].to(device)

# --- Run the model ---
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

def generate_data_chunk(N, indices, de, Inputs, batched_infer):
    
    """ Function to generate the training data for one instance in parallel. """
    Rand = torch.rand(N, 3 * N_perturbed).to(device)
    Rand_matrix = mat_generator_no_third(Rand, indices, (N, 3, height, width))
    d_at = de * Rand_matrix
    Inp = Inputs + d_at
    Inp_tensor = Inp.float()

    with torch.no_grad():
        out = batched_infer(Inp_tensor, model, batch_size=2)
    
    return out, Rand



t0 = time()



Inputs = Input.repeat(Nt,1,1,1)
Y, X = generate_data_chunk(Nt, indices, de, Inputs, batched_inference)


train_data_run_1 = time() - t0


# Y = Y.view(Y.shape[0], -1)
Y = Y.reshape(Y.shape[0], -1)


assert N_dir <= Nt, "Requested more samples than available!"

selected_indices = torch.randperm(Nt)[:N_dir]

X_dir = X[selected_indices, ...].to(device)
Y_dir = Y[selected_indices, ...].to(device)


# from Direction_training import compute_directions
def compute_f(A, X_batch):
    """
    Compute the function f(A) = (1 / (N * norm(A)^2)) * sum_i (A' * (X_i - mean(X)))^2
    """
    norm_A2 = torch.norm(A, p=2)
    mean_X = torch.mean(X_batch, dim=1, keepdim=True)  
    N = X_batch.shape[1]
    dX = X_batch - mean_X
    A_X = torch.matmul(A.T, dX)  # A' * (X - mean)
    
    # Compute covariance
    cov_batch = (1 / (norm_A2)) * torch.sqrt( torch.sum(torch.pow(A_X, 2)) / N )
    return cov_batch

# Function to perform the entire deflation process
def deflation(X, n, m, device, batch_size, Cov_prev):
    A_list = []  # To store all principal directions
    round = -1
    Largest = float('inf')

    while True:
        round += 1
        # Initialize A for this round
    
        if round>0:
            del X_batch
            del X_proj
            gc.collect()
    
        A = torch.randn(n, 1, requires_grad=True, device=device)  # Trainable parameter (n x 1)
    
        Initial_lr = 100*n
    
        optimizer = torch.optim.SGD([A], lr=Initial_lr)
    
        Cov_prev = float('inf')
    
        iter = 0
        Cov = compute_f(A, X)
        print(f"Initial guess for covaiance is: {Cov.item()}")
    
        # Train A to find the principal direction for this round
        while True:
            optimizer.zero_grad()  # Zero gradients
        
        
            # Sample a random batch from X
            indices = torch.randint(0, m, (batch_size,), device=device)  
            X_batch = X[:, indices]  # Select batch
        
            # Compute the function value f(A) and its gradients
            cov_batch = compute_f(A, X_batch)
        
            # Perform gradient ascent to maximize cov_batch
            (-cov_batch).backward(retain_graph=True)  # Equivalent to maximizing cov_batch
        
            optimizer.step()  # Update A
        
        
            # Check convergence
            if iter % 100 == 0:
                Cov = compute_f(A, X)
                print(f"Round {round+1} - Iteration {iter} - Covariance: {Cov.item()}")
            
                # Check if convergence condition is met
                if round == 0:
                    if abs(Cov - Cov_prev) < Cov_prev * 1e-2:
                        print(f"Convergence achieved at iteration {iter}. Stopping optimization.")
                        break
                else:
                    if abs(Cov - Cov_prev) < Largest * 1e-2:
                        print(f"Convergence achieved at iteration {iter}. Stopping optimization.")
                        break
                
            
                Cov_prev = Cov  # Update the previous covariance value
        
            iter += 1
    
        # Store the current principal direction A
        A = A / torch.norm( A , p=2 )
        A_list.append(A.clone().detach())
    
        # Remove the component of the data in the direction of A
        X_proj = torch.matmul(A.T, X) * A  # Project the data onto A
        del A
        print(X_proj.shape)  # Should be (64*84*11, 5*2000)
        X = X - X_proj  # Subtract projection from X to remove the direction
        X = X.detach()
    
    
        if round == 0:
            Largest = Cov
    
    
        if Cov<=Largest/100:
            print(f"A sufficient amount of principal directions ( {round+1} unit vectors) is collected.")
            break
        
    return A_list, X

def compute_directions(X , device, batch_size):

    # Send the tensor X to GPU
    X = X.to(device).T

    # Print the final shape
    print(X.shape)  # Should be (64*84*11, 5*2000)


    # Initialize the input data X and parameters A
    n = X.shape[0]
    m = X.shape[1]  # Number of data points

    # Set hyperparameters

    # Initialize Cov_prev with a very high value to start
    Cov_prev = float('inf')


    # Perform deflation
    t0 = time()
    A_list, X_updated = deflation(X, n, m, device, batch_size, Cov_prev)
    Direction_training_time = time() - t0
    # Print the final updated X and principal directions
    print("Updated X shape:", X_updated.shape)
    print("Principal directions found:")
    for i, A in enumerate(A_list):
        print(f"A{i+1}: {A}")
        
    return A_list , Direction_training_time

Directions, Direction_Training_time = compute_directions(Y_dir, device, 5)

Directions = torch.stack([d.squeeze(-1) for d in Directions])


C = 20 * (epsilon * Y.mean(dim=0) + (0.05 - epsilon) * 0.5 * (Y.min(dim=0).values + Y.max(dim=0).values))
dY = (Y - C.unsqueeze(0)).to(device)
dYV = dY @ Directions.T

torch.cuda.empty_cache()


# from Training_ReLU import Trainer_ReLU
def estimate_lipschitz(x, y, num_samples=1000):
    n = x.shape[0]
    print(n)
    slopes = []

    for _ in range(num_samples):
        i, j = random.sample(range(n), 2)
        diff_x = x[i] - x[j]
        diff_y = y[i] - y[j]

        norm_x = torch.norm(diff_x)
        norm_y = torch.norm(diff_y)

        if norm_x > 1e-8:  # avoid division by near-zero
            slope = norm_y / norm_x
            slopes.append(slope.item())

    return max(slopes)

def Trainer_ReLU(x , y , device, epochs, save_path):
    
    x = x.to(device)
    y = y.to(device)
    
    # Estimate λ before training
    lam = max( 10.0 , 5*estimate_lipschitz(x, y) )
    print(f"Estimated Lipschitz constant (empirical): {lam:.4f}")


    # Compute mean and std for normalization
    y_mean = y.mean(dim=0, keepdim=True)  # Shape [1, 10]
    y_std = y.std(dim=0, keepdim=True) + 1e-8  # Avoid division by zero

    # Normalize y
    y_norm = (y - y_mean) / y_std  # Shape [10000, 10]

    print(f"Using device: {device}")

    # Create DataLoader for mini-batch training
    batch_size = 20
    dataset = TensorDataset(x, y_norm)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define Neural Network Model with Two Hidden Layers
    class ReLUNetwork(nn.Module):
        def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
            super(ReLUNetwork, self).__init__()
            self.hidden1 = nn.Linear(input_dim, hidden_dim1)
            self.relu = nn.ReLU()
            self.hidden2 = nn.Linear(hidden_dim1, hidden_dim2)
            self.output = nn.Linear(hidden_dim2, output_dim)

        def forward(self, x):
            x = self.hidden1(x)
            x = self.relu(x)
            x = self.hidden2(x)
            x = self.relu(x)
            x = self.output(x)
            return x

    # Define Model and move to GPU
    input_dim   = x.shape[1]
    hidden_dim1 = input_dim
    output_dim  = y.shape[1]
    hidden_dim2 = output_dim
    model = ReLUNetwork(input_dim, hidden_dim1, hidden_dim2, output_dim).to(device)

    # Define Loss and Optimizer
    criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=0.01)


    # Training Loop
    t0 = time()
    for epoch in range(epochs):
        total_loss = 0
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {total_loss / len(dataloader):.4f}')
        
        
        if (epoch % 10 == 0)  and (epoch > epochs// 25):
            # === Enforce Lipschitz constraint per layer ===
            with torch.no_grad():
                for layer in [model.hidden1, model.hidden2, model.output]:
                    weight = layer.weight.data
                    norm = torch.linalg.norm(weight, ord=2)
                    scale = max(1.0, norm.item() / lam)
                    print(f'The scale is calculated as [{scale}]')
                    layer.weight.data = weight / scale

    # Extract trained parameters for Two hidden layer model
    W1 = model.hidden1.weight.detach().cpu().numpy()
    b1 = model.hidden1.bias.detach().cpu().numpy()
    W2 = model.hidden2.weight.detach().cpu().numpy()
    b2 = model.hidden2.bias.detach().cpu().numpy()
    W3 = model.output.weight.detach().cpu().numpy()
    b3 = model.output.bias.detach().cpu().numpy()

    # Reshape y_std to [10, 1] for correct broadcasting
    y_std_np = y_std.cpu().numpy().reshape(-1, 1)

    # Denormalize final layer (W3, b3)
    W3_denorm = (y_std_np * W3)
    b3_denorm = (y_std.cpu().numpy() * b3) + y_mean.cpu().numpy()

    # Save to MATLAB format
    scipy.io.savemat(save_path, {
        'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3': W3_denorm, 'b3': b3_denorm
    })

    print(f"Trained parameters saved to {save_path}")
    
    model.output.weight.data = torch.tensor(W3_denorm).to(device)
    model.output.bias.data = torch.tensor(b3_denorm).to(device)
    
    train_time = time() - t0
    
    return model , train_time

current_dir = os.getcwd()
save_path = os.path.join(current_dir, 'trained_relu_weights_2h_norm.mat')
epochs = 50
small_net, Model_training_time = Trainer_ReLU(X, dYV, device, epochs, save_path)

with torch.no_grad():
    pred = small_net(X) 

    approx_Y = pred @ Directions  + C.unsqueeze(0).to(device)  # shape: same as Y

t0 = time()
residuals = (Y.to(device) - approx_Y.to(device)).abs()
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
    
    Y_test_nc, X_test_nc = generate_data_chunk(curr_len, indices, de, Inputs, batched_inference)


    test_data_run.append(time() - t0)

    Y_test = Y_test_nc.reshape(Y_test_nc.shape[0], -1).to(device)
    del Y_test_nc

    t1 = time()
    pred = small_net(X_test_nc) @ Directions  + C.unsqueeze(0).to(device)  # shape: (dim2, curr_len)
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


matlab_code = fr"""


clear
clc

load('python_data.mat')
load('trained_relu_weights_2h_norm.mat')
Small_net.weights = {{double(W1) , double(W2), double(W3)}};
Small_net.biases = {{double(b1)' , double(b2)', double(b3)'}};

l1 = size(W1,1);
l2 = size(W2,1);
l0 = size(W1,2);

L = cell(l1,1);
L(:) = {{'poslin'}};
Small_net.layers{{1}} = L ;
L = cell(l2,1);
L(:) = {{'poslin'}};
Small_net.layers{{2}} = L ;

dim2 = {height}*{width}*{num_classes};
dimp = {height}*{width};

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

% addpath(genpath('C:\\Users\\navid\\Documents\\MATLAB\\MATLAB_prev\\others\\Files\\CDC2023\\Large_DNN\\src'))
addpath(genpath('~/13-ConformalInference/SSN_Reach/src'))
addpath(genpath('~/13-ConformalInference/nnv'))

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


Lb_pixels = reshape(Lb , [dimp, {num_classes}]);
Ub_pixels = reshape(Ub , [dimp, {num_classes}]);
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

mask = Lb_max <= Ub_pixels  # Shape: [720*960, 12], boolean

mask_np = mask.cpu().numpy()

classes = [[[] for _ in range(width)] for _ in range(height)]

for t in range(height * width):
    i = t % height
    j = t // height
    class_members = [k for k in range(12) if mask_np[t, k]]
    classes[i][j] = class_members


# img_tensor = torch.from_numpy(img).to(device)
img = Image.open(image_path).convert('RGB')
img_tensor = processor(images=img, return_tensors="pt")["pixel_values"]
output = batched_inference(img_tensor, model, batch_size=1)

output_np = output.squeeze().cpu().numpy() # shape: [19, 1024, 1024]
True_class_map = output_np.argmax(axis=0)  # shape: [1024, 1024]
True_class = [[int(True_class_map[i, j]) for j in range(width)] for i in range(height)]
# True_class = [[int(output_np[i, j] > 0) for j in range(width)] for i in range(height)]



# Initialize counters
robust = 0
nonrobust = 0
unknown = 0
attacked = N_perturbed  # Assuming this is defined

for i in range(height):
    for j in range(width):
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
dim_pic = height*width
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