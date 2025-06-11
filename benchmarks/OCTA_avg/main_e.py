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
<<<<<<< HEAD
from Provision import Reachability_provider
from Segmentation import Segmentor
=======
from SSN_provider import Reachability_provider
from SSN_classifier import classification
>>>>>>> 309924de7adbd067700e2848ca741849445904be




def OCTA_avg_exp1( start_loc, Epsi,  Nt, N_dir,
                  Ns, Nsp, rank, device,  threshold_normal,
                  sim_batch, trn_batch, epochs, surrogate_mode, dims):
        
    img = Image.open(image_name)
    to_tensor = transforms.ToTensor()
    img = to_tensor(img)


    ort_session = ort.InferenceSession('betti_avgpool.onnx', providers=['CUDAExecutionProvider'])
    img_tensor = img.reshape(1, 1, 304, 304).to(device, dtype=torch.float32)
    img_np = img_tensor.detach().cpu().numpy().astype(np.float32)
    output = ort_session.run(None, {'input': img_np})
    output = torch.tensor(output[0]).to(device)
    threshold = np.log(45/55)
    output_dim = output.shape
    output_np = output.squeeze().cpu().numpy()
    True_class = [[int(output_np[i, j] > threshold) for j in range(304)] for i in range(304)]



    indices = []

    _, _, H, W = img_np.shape
    for i in range(start_loc[0], H):
        for j in range(start_loc[1], W):
            indices.append([i, j])
            
    indices = np.array(indices)
    

    de  = 2 *  Epsi

    LB = img_tensor - Epsi


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
        'perturbation' : Epsi*255,
        'True_class' : True_class,
        'class_threshold' : threshold,
<<<<<<< HEAD
        'image_name' : image_name,
        'input_name' : 'input'
    }
    
    
    provide = Reachability_provider(
=======
        'image_name' : image_name
    }
    
    
    provider = Reachability_provider(
>>>>>>> 309924de7adbd067700e2848ca741849445904be
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
    
<<<<<<< HEAD
    provide.Provider()
=======
    provider.Provider()
>>>>>>> 309924de7adbd067700e2848ca741849445904be




def OCTA_avg_exp2( projection_batch, guarantee, device, src_dir, nnv_dir ):
    
    params = {
        'projection_batch' : projection_batch,
        'guarantee': guarantee,
    }
    
<<<<<<< HEAD
    Segment = Segmentor(
=======
    classifier = classification(
>>>>>>> 309924de7adbd067700e2848ca741849445904be
        device = device,
        src_dir = src_dir,
        nnv_dir = nnv_dir,
        params = params
        )

<<<<<<< HEAD
    Segment.Mask_titles()
=======
    classifier.Mask_titles()
>>>>>>> 309924de7adbd067700e2848ca741849445904be





if __name__ == '__main__':
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    start_loc = (0, 0)
    Ns = 8000
    Nsp = 200
    rank = 7999
    guarantee = 0.999
    delta_rgb = 3
    de = delta_rgb / 255.0
    Nt = 1500
    N_dir = 200
    threshold_normal = 1e-5
    sim_batch = 50
    trn_batch = 20
    epochs = 400
    surrogate_mode = 'ReLU'
    src_dir = os.path.join(root_dir, 'src')
    nnv_dir = '/home/hashemn/nnv'
    # dims = [ 100 , 'auto' ]
    dims = [6, 6]
    projection_batch = 500
    
    if not os.path.isdir(nnv_dir):
        sys.exit(f"❌ Error: NNV directory not found at '{nnv_dir}'.\n"
                 f"Please check the path and ensure NNV is properly installed.")

    print(f"✅ NNV directory found at: {nnv_dir}")


    image_names = [
        '10491.bmp',
        ]

    
    Epsi_list = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01]
    
    ii=0
    for idx, image_name in enumerate(image_names):
        for Epsi in Epsi_list:
            
            print(f"Running: {image_name} with N_perturbed= ALL")
            
            ii = ii+1
            OCTA_avg_exp1( start_loc, Epsi/255,  Nt, N_dir,
                           Ns, Nsp, rank, device,  threshold_normal,
                           sim_batch, trn_batch, epochs, surrogate_mode, dims)
            
            
            OCTA_avg_exp2( projection_batch, guarantee, device, src_dir, nnv_dir )