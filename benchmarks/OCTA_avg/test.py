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
from PIL import Image
from torchvision import transforms
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
root_dir = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(root_dir)
from utils import plot_binary_logits_to_mask


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
Nt = 10
N_dir = 6
threshold_normal = 1e-5
sim_batch = 2
trn_batch = 5
epochs = 50
surrogate_mode = 'ReLU'
src_dir = os.path.join(root_dir, 'src')

nnv_dir = '/home/hashemn/nnv'

if not os.path.isdir(nnv_dir):
    sys.exit(f"❌ Error: NNV directory not found at '{nnv_dir}'.\n"
             f"Please check the path and ensure NNV is properly installed.")

print(f"✅ NNV directory found at: {nnv_dir}")


current_dir = os.getcwd()


img = Image.open('10491.bmp')
to_tensor = transforms.ToTensor()
img = to_tensor(img)


ort_session = ort.InferenceSession('betti_avgpool.onnx', providers=['CUDAExecutionProvider'])
img_tensor = img.reshape(1, 1, 304, 304).to(device, dtype=torch.float32)
img_np = img_tensor.detach().cpu().numpy().astype(np.float32)
output = ort_session.run(None, {'input': img_np})
output = torch.tensor(output[0]).to(device)
threshold = np.log(45/55)
plot_binary_logits_to_mask(output,threshold)
output_dim = output.shape
output_np = output.squeeze().cpu().numpy()
True_class = [[int(output_np[i, j] > threshold) for j in range(304)] for i in range(304)]



indices = []

_, _, H, W = img_np.shape
for i in range(start_loc[0], H):
    for j in range(start_loc[1], W):
        indices.append([i, j])

indices = np.array(indices)



Epsi = 0.0001/255

de  = 2 *  Epsi

LB = img_tensor - Epsi


dims =[100, 'auto']


params = {
    'N_perturbed' : len(indices),
    'delta_rgb' : 1,
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
    'dims' : dims
}
analyzer = ReachabilityAnalyzer(
    True_class = True_class,
    model = ort_session,
    image_name = '10491.bmp',
    LB = LB,
    de = de,
    indices = indices,
    original_dim = (1, 304, 304),
    output_dim = output_dim,
    device=device,
    mode = surrogate_mode,
    class_threshold = 0,
    src_dir = src_dir,
    nnv_dir = nnv_dir,
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
