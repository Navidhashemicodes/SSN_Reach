# -*- coding: utf-8 -*-
"""
Created on Wed May 21 23:57:52 2025

@author: navid
"""

import torch
import numpy as np
import onnxruntime as ort
import cv2
from time import time
import os
import sys
from tqdm import tqdm
from scipy.stats import beta
from torch.cuda.amp import autocast
import matlab.engine
import scipy.io
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
root_dir = pathlib.Path(__file__).resolve().parents[2]
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
    


image_name = 'CHNCXR_0005_0.png'
model_name = 'lung_segmentation.onnx'
print(f"Running: {image_name} with N_perturbed={N_perturbed}")


current_dir = os.getcwd()
model_path = os.path.join(current_dir, 'models', model_name)
image_path = os.path.join(current_dir, 'images', image_name)

ort_session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])

img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # ensures it's grayscale
img = cv2.resize(img, (512, 512))
img = img.astype(np.float32) / 255.0
img = img.reshape(1, 1, 512, 512)
at_im = img.copy()


img_tensor = torch.from_numpy(img).to(device)
x = img_tensor.to(torch.float16)  # Use half precision
x_numpy = x.cpu().numpy().astype(np.float32)
output = ort_session.run(None, {'input': x_numpy})
output = torch.tensor(output[0]).to(device)

output_dim = output.shape

output_np = output.squeeze().cpu().numpy()  # shape: [512, 512]

True_class = [[int(output_np[i, j] > 0) for j in range(512)] for i in range(512)]

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
    original_dim = (1, 512, 512),
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
