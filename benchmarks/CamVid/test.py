# -*- coding: utf-8 -*-
"""
Created on Wed May 21 23:57:52 2025

@author: navid
"""

import torch
import numpy as np
import onnxruntime as ort
import cv2
import os
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
root_dir = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(root_dir)

from utils import plot_logits_to_mask

reach_factory_path = os.path.join(root_dir, 'Reach_Factory')
sys.path.append(reach_factory_path)

from Reach4SSN import ReachabilityAnalyzer


# The following hyperparameters can be adjusted to fit the verification process
# within the limits of your GPU memory. The current values are intentionally kept
# small for compatibility with GPUs with limited memory.

# Ns: Number of calibration samples. Increasing Ns can improve the level of formal guarantees.

# Nsp: Number of calibration samples processed per iteration on the GPU. Adjust this value
# to ensure that the per-iteration data fits in memory. Higher values reduce runtime,
# but increase GPU memory requirements.

# Nt: Number of training samples to be loaded onto the GPU in full. A larger Nt typically
# leads to tighter (less conservative) bounds, but requires more memory. On GPUs with
# limited memory, you may need to reduce Nt, accepting slightly more conservative results.


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
start_loc = (0, 0)
Ns = 10
Nsp = 2
rank = 7
guarantee = 0.999
delta_rgb = 150
de = delta_rgb / 255.0
Nt = 10
N_dir = 6
threshold_normal = 1e-5
sim_batch = 2
trn_batch = 5
epochs = 50
N_perturbed = 17
surrogate_mode = 'ReLU'
src_dir = os.path.join(root_dir, 'src')
    


image_name = '0001TP_008790.png'
model_name = 'BiSeNet.onnx'
print(f"Running: {image_name} with N_perturbed={N_perturbed}")


current_dir = os.getcwd()
model_path = os.path.join(current_dir, 'models', model_name)
image_path = os.path.join(current_dir, 'images', image_name)

ort_session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])

img = cv2.imread(image_path)  # BGR format
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (960, 720))  # Ensure correct input size
img_np = img.astype(np.float32) / 255.0  # [H, W, 3] float32 in [0, 1]
img_np = np.transpose(img_np, (2, 0, 1))  # [3, 720, 960]
img_np = img_np.reshape(1, 3, 720, 960)

# Make a copy for perturbation
at_im = img_np.copy()  # Also shape: [1, 3, 720, 960]

# --- Perturbation loop ---
ct = 0
indices = []
_, _, H, W = img_np.shape     # Shape: [1, 3, 720, 960]

for i in range(start_loc[0], H):
    for j in range(start_loc[1], W):
        if np.min(img_np[0, :, i, j]) > 150 / 255.0:
            at_im[0, :, i, j] = 0.0  # Reset all 3 channels at once
            indices.append([i, j])
            ct += 1
            if ct == N_perturbed:
                print(f"{N_perturbed} pixels found.")
                break
    if ct == N_perturbed:
        break

indices = np.array(indices)



# --- Normalize image ---
mean_vals = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
std_vals = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)

at_im_norm = (at_im - mean_vals) / std_vals


img_norm = (img_np - mean_vals) / std_vals
img_tensor = torch.from_numpy(img_norm).to(device)
x = img_tensor.to(torch.float16)  # Use half precision
x_numpy = x.cpu().numpy().astype(np.float32)
output = ort_session.run(None, {'input': x_numpy})
output = torch.tensor(output[0]).to(device)


plot_logits_to_mask(output)

output_dim = output.shape

_, True_class_tensor = torch.max(output, dim=1)  # shape: (1, 720, 960)
True_class_tensor = True_class_tensor.squeeze(0)  # shape: (720, 960)
True_class = True_class_tensor.cpu().tolist()


at_im_tensor = torch.from_numpy(at_im_norm).to(device)
params = {
    'N_perturbed' : N_perturbed,
    'de' : de,
    'image_name' : image_name,
    'Nt' : Nt,
    'N_dir' : N_dir,
    'Ns' : Ns,
    'Nsp' : Nsp,
    'rank' : rank,
    'guarantee': guarantee,
    'threshold_normal' : threshold_normal,
    'trn_batch' : trn_batch,
    'sim_batch' : sim_batch,
    'epochs' : epochs,
    'device' : device,
}
analyzer = ReachabilityAnalyzer(
    True_class = True_class,
    model = ort_session,
    image_name = image_name,
    LB = at_im_tensor,
    de = de,
    indices = indices,
    original_dim = (3, 720, 960),
    output_dim = output_dim,
    device=device,
    mode = surrogate_mode,
    class_threshold = 0,
    src_dir = src_dir,
    params=params
)
analyzer.Mask_titles()

remove_path =  os.path.join(current_dir, 'Matlab_data.mat')
os.remove(remove_path)
remove_path =  os.path.join(current_dir, 'python_data.mat')
os.remove(remove_path)
if surrogate_mode == 'ReLU':
    remove_path =  os.path.join(current_dir, 'trained_relu_weights_2h_norm.mat')
    os.remove(remove_path)
