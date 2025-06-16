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
from torchvision import transforms
import gc
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
root_dir = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(root_dir)
from utils import plot_logits_to_mask


reach_factory_path = os.path.join(root_dir, 'Reach_Factory')
sys.path.append(reach_factory_path)
from SSN_provider import Reachability_provider
from SSN_classifier import classification




def Camvid_exp1( start_loc, N_perturbed, delta_rgn,  Nt, N_dir,
                  Ns, Nsp, rank, device,  threshold_normal,
                  sim_batch, trn_batch, epochs, surrogate_mode, dims):
    
    
    de = delta_rgb / 255.0
    model_name = 'BiSeNet.onnx'

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
    
    # plot_logits_to_mask(output)

    output_dim = output.shape

    _, True_class_tensor = torch.max(output, dim=1)  # shape: (1, 720, 960)
    True_class_tensor = True_class_tensor.squeeze(0)  # shape: (720, 960)
    True_class = True_class_tensor.cpu().tolist()


    at_im_tensor = torch.from_numpy(at_im_norm).to(device)

    
    params = {
        'sim_batch' : sim_batch,
        'Nt' : Nt,
        'N_dir' : N_dir,
        'trn_batch' : trn_batch,
        'dims' : dims,
        'epochs' : epochs,
        'threshold_normal' : threshold_normal,
        'Ns' : Ns,
        'Nsp' : Nsp,
        'rank' : rank,
        'perturbation' : delta_rgb,
        'True_class' : True_class,
        'class_threshold' : None,
        'image_name' : image_name
    }
    
    
    provider = Reachability_provider(
        de = torch.tensor(de/std_vals).to(device),
        indices = indices,
        device = device,
        model = ort_session,
        LB = at_im_tensor,
        original_dim = (3, 720, 960),
        output_dim = output_dim,
        mode = surrogate_mode,
        params = params
        )
    
    provider.Provider()




def Camvid_exp2( projection_batch, guarantee, device, src_dir, nnv_dir ):
    
    params = {
        'projection_batch' : projection_batch,
        'guarantee': guarantee,
    }
    
    classifier = classification(
        device = device,
        src_dir = src_dir,
        nnv_dir = nnv_dir,
        params = params
        )

    classifier.Mask_titles()





if __name__ == '__main__':
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    start_loc = (0, 0)
    Ns = 8000
    Nsp =  150
    rank = 7999
    guarantee = 0.999
    Nt = 2100
    N_dir = 150
    threshold_normal = 1e-5
    sim_batch = 5
    trn_batch = 10
    epochs = 200
    surrogate_mode = 'ReLU'
    src_dir = os.path.join(root_dir, 'src')
    nnv_dir = '/home/hashemn/nnv'
    projection_batch = 720*960
    dims = ['auto' , 'auto']
    
    
    if not os.path.isdir(nnv_dir):
        sys.exit(f"❌ Error: NNV directory not found at '{nnv_dir}'.\n"
                 f"Please check the path and ensure NNV is properly installed.")

    print(f"✅ NNV directory found at: {nnv_dir}")


    image_names = [
        '0001TP_008790.png',
        '0001TP_010110.png',
        '0001TP_010290.png',
        '0006R0_f02190.png',
        'Seq05VD_f00000.png',
        'Seq05VD_f00090.png',
        'Seq05VD_f01320.png',
        'Seq05VD_f01410.png',
        'Seq05VD_f02070.png',
        'Seq05VD_f03540.png'
        ]

    
    delta_rgb = 3

    N_perturbed_list = [ 34, 51, 68, 85, 102]
    
    ii=0
    for idx, image_name in enumerate(image_names):
        for N_perturbed in N_perturbed_list:
            
            print(f"Running: {image_name} with N_perturbed={N_perturbed}")
            
            Camvid_exp1( start_loc, N_perturbed, delta_rgb,  Nt, N_dir,
                           Ns, Nsp, rank, device,  threshold_normal,
                           sim_batch, trn_batch, epochs, surrogate_mode, dims)
            
            
            Camvid_exp2( projection_batch, guarantee, device, src_dir, nnv_dir )
            
            gc.collect()
            torch.cuda.empty_cache()