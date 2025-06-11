# -*- coding: utf-8 -*-
"""
Created on Wed May 21 23:57:52 2025

@author: navid
"""

import torch
import numpy as np
import onnxruntime as ort
import cv2
from PIL import Image
import os
import sys
import pathlib
import torchvision.transforms as transforms

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
root_dir = pathlib.Path(__file__).resolve().parents[2]
reach_factory_path = os.path.join(root_dir, 'Reach_Factory')
sys.path.append(reach_factory_path)
from Reach4SSN import ReachabilityAnalyzer




def CheXpert_exp( start_loc, N_perturbed, delta_rgb, image_name, Nt, N_dir,
                  Ns, Nsp, rank, guarantee, device,  threshold_normal,
                  sim_batch, trn_batch, epochs, surrogate_mode, src_dir, nnv_dir, dims):
        
    model_name = 'lung_segmentation.onnx'

    current_dir = os.getcwd()
    model_path = os.path.join(current_dir, 'models', model_name)
    image_path = os.path.join(current_dir, 'images', image_name)

    ort_session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])

    eval_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])


    img = Image.open(image_path).convert('L')  # 'L' mode = single-channel grayscale
    img = eval_transforms(img)
    img_tensor = img.unsqueeze(0).to(device)
    img_np = img_tensor.detach().cpu().numpy().astype(np.float32)

    at_im = img_np.copy()
    
    
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

    _, _, H, W = img_np.shape
    for i in range(start_loc[0], H):
        for j in range(start_loc[1], W):
            if np.min(img_np[:,:,i, j]) > 150 / 255.0:
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
        'N_perturbed' : len(indices),
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
        'dims' : dims
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
        nnv_dir = nnv_dir,
        params=params
    )
    analyzer.Mask_titles()

    remove_path =  os.path.join(current_dir, 'Matlab_data.mat')
    os.remove(remove_path)
    
    path1 = os.path.join(current_dir, 'python_data.mat')
    path2 = os.path.join(current_dir, 'python_data.npz')
    if os.path.exists(path1):
        os.remove(path1)
    elif os.path.exists(path2):
        os.remove(path2)
        
        
    if surrogate_mode == 'ReLU':
        remove_path =  os.path.join(current_dir, 'trained_relu_weights_2h_norm.mat')
        os.remove(remove_path)




if __name__ == '__main__':
    
    # The following hyperparameters can be adjusted to fit the verification process
    # within the limits of your GPU memory. The current values are according to experiments
    # in the submission

    # Ns: Number of calibration samples. Increasing Ns can improve the level of formal guarantees.

    # Nsp: Number of calibration samples processed per iteration on the GPU. Adjust this value
    # to ensure that the per-iteration data fits in memory. Higher values reduce runtime,
    # but increase GPU memory requirements.

    # Nt: Number of training samples to be loaded onto the GPU in full. A larger Nt typically
    # leads to tighter (less conservative) bounds, but requires more memory. On GPUs with
    # limited memory, you may need to reduce Nt, accepting slightly more conservative results.
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    start_loc = (0, 0)
    Ns = 8000
    Nsp = 1000
    rank = 8000
    guarantee = 0.999
    delta_rgb = 3
    de = delta_rgb / 255.0
    Nt = 1500
    N_dir = 100
    threshold_normal = 1e-5
    sim_batch = 50
    trn_batch = 20
    epochs = 200
    surrogate_mode = 'ReLU'
    src_dir = os.path.join(root_dir, 'src')
    nnv_dir = '/home/hashemn/nnv'
    dims = ['auto' , 'auto']
    
    if not os.path.isdir(nnv_dir):
        sys.exit(f"❌ Error: NNV directory not found at '{nnv_dir}'.\n"
                 f"Please check the path and ensure NNV is properly installed.")

    print(f"✅ NNV directory found at: {nnv_dir}")


    image_names = [
        'CHNCXR_0005_0.png',
        'MCUCXR_0258_1.png',
        'MCUCXR_0264_1.png',
        'MCUCXR_0266_1.png',
        'MCUCXR_0275_1.png',
        'MCUCXR_0282_1.png',
        'MCUCXR_0289_1.png',
        'MCUCXR_0294_1.png',
        'MCUCXR_0301_1.png',
        'MCUCXR_0309_1.png',
        'MCUCXR_0311_1.png',
        'MCUCXR_0313_1.png',
        'MCUCXR_0316_1.png',
        'MCUCXR_0331_1.png',
        'MCUCXR_0334_1.png',
        'MCUCXR_0338_1.png',
        'MCUCXR_0348_1.png',
        'MCUCXR_0350_1.png',
        'MCUCXR_0352_1.png',
        'MCUCXR_0354_1.png'
        ]

    N_perturbed_list = [17, 34, 51, 68, 85, 102]
    # N_perturbed_list = [17, 34]
    
    ii=0
    for idx, image_name in enumerate(image_names):
        for N_perturbed in N_perturbed_list:
            
            print(f"Running: {image_name} with N_perturbed={N_perturbed}")
            
            ii = ii+1
            CheXpert_exp( start_loc, N_perturbed, delta_rgb, image_name, Nt, N_dir,
                          Ns, Nsp, rank, guarantee, device,  threshold_normal,
                          sim_batch, trn_batch, epochs, surrogate_mode, src_dir, nnv_dir, dims)
