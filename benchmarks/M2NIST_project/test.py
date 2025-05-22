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
import scipy.io


sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
root_dir = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(root_dir)

from utils import plot_logits_to_mask

reach_factory_path = os.path.join(root_dir, 'Reach_Factory')
sys.path.append(reach_factory_path)

from Reach4SSN import ReachabilityAnalyzer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
start_loc = (0, 0)
Ns = 10
Nsp = 2
rank = 7
guarantee = 0.999
delta_rgb = 150
de = delta_rgb
Nt = 10
N_dir = 6
threshold_normal = 1e-5
sim_batch = 2
trn_batch = 5
epochs = 50
N_perturbed = 17
surrogate_mode = 'ReLU'
src_dir = os.path.join(root_dir, 'src')


image_name = 'm2nist_6484_test_images.mat'
model_name = 'm2nist_dilated_72iou_24layer.onnx'
print(f"Running: {image_name} with N_perturbed={N_perturbed}")
current_dir = os.getcwd()
model_path = os.path.join(current_dir, 'models', model_name)
image_path = os.path.join(current_dir, 'images', image_name)


# Load MATLAB data
mat = scipy.io.loadmat(image_path)
images = mat['im_data']  # (64, 84, 1000)
image = images[:,:,0]
image = image[:, :, np.newaxis]

# Convert shape and type
image_np = np.transpose(image, (2, 0, 1))      # (1, 64, 84)
image_np = image_np[:, np.newaxis, :, :].astype(np.float32)  # (1, 1, 64, 84)


ort_session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])


at_im = image_np.copy()  # Also shape: [1, 3, 720, 960]

ct = 0
indices = []
_, _, H, W = image_np.shape     # Shape: [1, 3, 720, 960]

for i in range(start_loc[0], H):
    for j in range(start_loc[1], W):
        if np.min(image_np[0, :, i, j]) > 150:
            at_im[0, :, i, j] = 0.0  # Reset all 3 channels at once
            indices.append([i, j])
            ct += 1
            if ct == N_perturbed:
                print(f"{N_perturbed} pixels found.")
                break
    if ct == N_perturbed:
        break

indices = np.array(indices)


image_tensor = torch.from_numpy(image_np).to(device)
x = image_tensor.to(torch.float16)  # Use half precision
x_numpy = x.cpu().numpy().astype(np.float32)
output = ort_session.run(None, {'input': x_numpy})
output = torch.tensor(output[0]).to(device)


plot_logits_to_mask(output)

output_dim = output.shape

_, True_class_tensor = torch.max(output, dim=1)  # shape: (1, 720, 960)
True_class_tensor = True_class_tensor.squeeze(0)  # shape: (720, 960)
True_class = True_class_tensor.cpu().tolist()


at_im_tensor = torch.from_numpy(at_im).to(device)
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
    original_dim = (1, 64, 84),
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
