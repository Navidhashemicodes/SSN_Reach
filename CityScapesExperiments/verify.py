import os
import sys
import torch
import numpy as np
from PIL import Image
# from tqdm import tqdm
from scipy.stats import beta
from time import time
from joblib import Parallel, delayed
# import matlab.engine
import scipy.io

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


"""Command-Line Arguments

[1] <config_name>: path to a config file

Example usage:

python pyversion.py --config config.yaml
"""

torch.manual_seed(0)

def load_config(config_path="config.yaml"):
    """Load config file"""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    return parser.parse_args()


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
N = config['N'] # total number of samples for learning the principal directions (train the surrogate model)
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
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b2-finetuned-cityscapes-1024-1024")
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

# --- Apply darkening attack
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
at_im_tensor = processor(images=at_pil, return_tensors="pt")["pixel_values"]

# --- Run the model ---
with torch.no_grad():
    if model_name == "hrnet":
        output = model(at_im_tensor)  # [B, C, H, W]
        # pred = output.argmax(1).squeeze(0).cpu().numpy()  # [H, W]
        # pred = F.interpolate(output, size=(512, 1024), mode='bilinear', align_corners=True)
        predicted_classes = F.interpolate(output, size=(1024, 2048), mode='bilinear', align_corners=True)
        predicted_classes = torch.argmax(predicted_classes, dim=1).squeeze(0)  # shape: (512, 1024)

    elif model_name == "segformer":
        outputs = model(at_im_tensor)  # This will now be [1, 3, 1024, 1024]
        logits = outputs.logits  # shape: [1, 19, 128, 128]

        # Upsample to 1024x1024
        logits = F.interpolate(logits, size=(1024, 1024), mode='bilinear', align_corners=False)
        predicted_classes = torch.argmax(logits, dim=1).squeeze().cpu().numpy()  # shape: [1024, 1024]

    else: # model_name == "mask2former"
        outputs = model(at_im_tensor) # this might break...
        predicted_classes = processor.post_process_semantic_segmentation(outputs, target_sizes=[(1024, 1024)])[0]
        predicted_classes = predicted_classes.cpu().numpy()
        # gt = label.squeeze(0).numpy()  # ground truth (H, W)

    gt = label.squeeze().numpy()

print("Inference complete. Predicted shape:", predicted_classes.shape)


# --- Output ---
# TODO:

img_shape = (height, width, 3)
std_vals = torch.tensor([0.229, 0.224, 0.225])

def mat_generator_no_third(values, indices, original_dim):
    # TODO:
    """Need to edit this to support batch computation"""
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

def generate_data_chunk(i, indices, de, std_vals, Input, model):
    # TODO:
    """What is this???""" 
    """Need to edit this to support batch computation"""
    """Look up vmap... this could come in handy"""
    # d_at is the amount of perturbation we add to the already darkening attacked image
    Rand = torch.rand(3 * N_perturbed)
    Rand_matrix = mat_generator_no_third(Rand, indices, (height, width, 3))
    d_at = torch.zeros((height, width, 3))
    for c in range(3):
        d_at[:, :, c] = de * Rand_matrix[:, :, c] / std_vals[c]
    d_at_tensor = d_at.permute(2, 0, 1).unsqueeze(0) 
    Inp = Input + d_at_tensor
    # Inp_tensor = Inp.permute(2, 0, 1).unsqueeze(0).float()
    Inp_tensor = Inp.float()

    # Direct BiSeNet inference (model is already preloaded)
    with torch.no_grad():
        out = model(Inp_tensor)  # (1, 3, 720, 960) -> (1, 12, 720, 960)

    """AttributeError: 'SemanticSegmenterOutput' object has no attribute 'squeeze'... 
    
    TODO: In other words, I need to address this with cases depending on what model we're using...
    """

    if model_name == "mask2former" or "segformer":
        logits = out.logits
        out = torch.argmax(logits, dim=1)
    else:
        out = out.squeeze(0).permute(1, 2, 0)  # (720, 960, 12)
    
    return out, Rand

# initialize Y and X
# Y = torch.zeros((height, width, num_classes, N)) # ???
# X = torch.zeros((3 * N_perturbed, N))

X = torch.zeros((3 * N_perturbed, N))
if model_name == "segformer":
    Y = torch.zeros((256, 256, num_classes, N))
elif model_name == "mask2former":
    pass
else:
    pass

# start timing
t0 = time()

results = Parallel(n_jobs=10, backend='loky', verbose=10)(
        delayed(generate_data_chunk)(i, indices, de, std_vals, Input, model) for i in range(N))



# collect results and assign them
for i, (out, Rand) in enumerate(results):
    Y[:, :, :, i] = out
    X[:, i] = Rand

Y = processor.post_process_semantic_segmentation(Y, target_sizes=[(1024, 1024)])[0]

train_data_run_1 = time() - t0

Y = Y.view(-1, Y.shape[-1])

assert N_dir <= N, "Requested more samples than available!"

selected_indices = torch.randperm(N)[:N_dir]

# select and move to GPU
X_dir = X[:, selected_indices].to(device)
Y_dir = Y[:, selected_indices].to(device)


from Direction_training import compute_directions

Directions, Direction_Training_time = compute_directions(Y_dir, device)

Directions = torch.stack([d.squeeze(-1) for d in Directions]).cpu()

# compute stats of Y on CPU
min_Y, _ = Y.min(dim=1, keepdim=True)
max_Y, _ = Y.max(dim=1, keepdim=True)
mean_Y = Y.mean(dim=1, keepdim=True)

# compute C on CPU, then move to GPU
C = 20.0 * (0.01 * mean_Y + (0.05 - 0.01) * 0.5 * (min_Y + max_Y))

dYV = Directions @ (Y - C)

torch.cuda.empty_cache()


from Training_ReLU import Trainer_ReLU

current_dir = os.getcwd()
save_path = os.path.join(current_dir, 'trained_relu_weights_2h_norm.mat')
epochs = 500 # TODO: make this a CLI arg?
small_net, Model_training_time = Trainer_ReLU(X, dYV, device, epochs, save_path)
small_net = small_net.cpu()

with torch.no_grad():
    pred = small_net(X.T).T

    approx_Y = Directions.T @ pred + C

t0 = time()
residuals = (Y - approx_Y).abs()
threshold_normal = 1e-5
res_max = residuals.max(dim=1).values
res_max[res_max < threshold_normal] = threshold_normal

trn_time1 = time() - t0

ell=  rank
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
    torch.manual_seed(nc + 1) # why are we changing seed?

    Y_test_nc = torch.zeros((height, width, num_classes, curr_len))
    X_test_nc = torch.zeros((3 * N_perturbed, curr_len))

    t0 = time()

    results = Parallel(n_jobs=10, backend='loky', verbose=10)(
            delayed(generate_data_chunk)(i, indices, de, std_vals, Input, model) for i in range(Ns))

    for i, (out, Rand) in enumerate(results):
        Y_test_nc[:, :, :, i] = out
        X_test_nc[:, i] = Rand

    test_data_run.append(time() - t0)

    Y_test = Y_test_nc.reshape((-1, curr_len))
    del Y_test_nc

    t1 = time()
    pred = Directions.T @ small_net(X_test_nc.T).T + C
    res_tst = (Y_test - pred).abs()
    Rs[ind:ind + curr_len] = torch.max(res_tst / res_max[:, None], dim=0).values
    res_test_time.append(time() - t1)

    del Y_test, X_test_nc, res_tst
    ind += curr_len


t0 = time()

with torch.no_grad():
    Rs_sorted = torch.sort(Rs).values
    R_star = Rs_sorted[ell]
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

# --- Start MATLAB ---
eng = matlab.engine.start_matlab()

# TODO: can I do string interpolation here?
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

mask = Lb_max <= Ub_pixels  # Shape: [720*960, 12], boolean

mask_np = mask.cpu().numpy()

classes = [[[] for _ in range(width)] for _ in range(height)]

for t in range(height * width):
    i = t % height
    j = t % width
    class_members = [k for k in range(num_classes) if mask_np[t, k]]


img = Image.open(image_path).convert('RGB')
img_np = np.array(img).astype(np.float32) / 255.0  # shape: (H, W, 3), range [0, 1]

img_norm = (img_np - mean_vals) / std_vals

img_tensor = img_norm.permute(2, 0, 1).unsqueeze(0).float()
output = model(img_tensor)  # Assuming the model is already loaded

True_class = {}
for i in range(height):
    for j in range(width):
        _, True_class[(i, j)] = torch.max(output[0, :, i, j], 0)  # Assuming output shape is [1, C, H, W]

# Initialize counters
robust = 0
nonrobust = 0
unknown = 0
attacked = N_perturbed  # Assuming this is defined

for i in range(height):
    for j in range(width):
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
dim_pic = height * width 
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




#

