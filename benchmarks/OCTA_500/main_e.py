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

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
root_dir = pathlib.Path(__file__).resolve().parents[2]
reach_factory_path = os.path.join(root_dir, 'Reach_Factory')
sys.path.append(reach_factory_path)
from Provision import Reachability_provider
from Segmentation import Segmentor




def OCTA_max_exp1( start_loc, N_perturbed, delta_rgn,  Nt, N_dir,
                  Ns, Nsp, rank, device,  threshold_normal,
                  sim_batch, trn_batch, epochs, surrogate_mode, dims):
    
    
    model_name = 'betti_best.onnx'

    current_dir = os.getcwd()
    model_path = os.path.join(current_dir, 'models', model_name)
    image_path = os.path.join(current_dir, 'images', image_name)


    img = Image.open(image_path)
    to_tensor = transforms.ToTensor()
    img = to_tensor(img)


    ort_session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])

    img_tensor = img.reshape(1, 1, 304, 304).to(device, dtype=torch.float32)
    img_np = img_tensor.detach().cpu().numpy().astype(np.float32)
    at_im = img_np.copy()
    
    output = ort_session.run(None, {'input': img_np})
    output = torch.tensor(output[0]).to(device)

    output_dim = output.shape

    # output_np = torch.sigmoid(output).squeeze().cpu().numpy()  # shape: [512, 512]
    output_np = output.squeeze().cpu().numpy()
    threshold = np.log(45/55)
    True_class = [[int(output_np[i, j] > threshold) for j in range(304)] for i in range(304)]

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
    LB = torch.from_numpy(at_im).to(device)
    de = delta_rgb / 255
    
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
        'class_threshold' : threshold,
        'image_name' : image_name,
        'input_name' : 'input'
    }
    
    
    provide = Reachability_provider(
        de = de,
        indices = indices,
        device = device,
        model = ort_session,
        LB = LB,
        original_dim = (1, 304, 304),
        output_dim = output_dim,
        mode = surrogate_mode,
        params = params
        )
    
    provide.Provider()




def OCTA_max_exp2( projection_batch, guarantee, device, src_dir, nnv_dir ):
    
    params = {
        'projection_batch' : projection_batch,
        'guarantee': guarantee,
    }
    
    Segment = Segmentor(
        device = device,
        src_dir = src_dir,
        nnv_dir = nnv_dir,
        params = params
        )

    Segment.Mask_titles()





if __name__ == '__main__':
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    start_loc = (0, 0)
    Ns = 8000
    Nsp = 200
    rank = 7999
    guarantee = 0.999
    Nt = 1500
    N_dir = 100
    threshold_normal = 1e-5
    sim_batch = 50
    trn_batch = 20
    epochs = 200
    surrogate_mode = 'ReLU'
    src_dir = os.path.join(root_dir, 'src')
    nnv_dir = '/home/hashemn/nnv'
    projection_batch = 304*304
    dims = ['auto' , 'auto']
    
    if not os.path.isdir(nnv_dir):
        sys.exit(f"❌ Error: NNV directory not found at '{nnv_dir}'.\n"
                 f"Please check the path and ensure NNV is properly installed.")

    print(f"✅ NNV directory found at: {nnv_dir}")


    image_names = [
        '10491.bmp',
        '10305.bmp',
        '10395.bmp',
        '10495.bmp',
        '10301.bmp',
        '10401.bmp',
        '10372.bmp',
        '10425.bmp',
        '10439.bmp',
        '10418.bmp',
        '10399.bmp',
        '10469.bmp',
        '10323.bmp',
        '10382.bmp',
        '10486.bmp',
        '10302.bmp',
        '10499.bmp',
        '10444.bmp',
        '10343.bmp',
        '10367.bmp'
        ]

    
    N_perturbed = 102
    
    delta_rgb_list = [10, 30, 60, 90, 120, 150] 
    

    for idx, image_name in enumerate(image_names):
        for delta_rgb in delta_rgb_list:
            
            print(f"Running: {image_name} with N_perturbed= ALL")
            
            OCTA_max_exp1( start_loc, N_perturbed, delta_rgb,  Nt, N_dir,
                           Ns, Nsp, rank, device,  threshold_normal,
                           sim_batch, trn_batch, epochs, surrogate_mode, dims)
            
            
            OCTA_max_exp2( projection_batch, guarantee, device, src_dir, nnv_dir )