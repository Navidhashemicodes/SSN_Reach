#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 15:41:10 2025

@author: hashemn
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
from SSN_provider import Reachability_provider
from SSN_classifier import classification


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
    projection_batch = 720*960*12
    dims = ['auto' , 'auto']
    
    
    if not os.path.isdir(nnv_dir):
        sys.exit(f"❌ Error: NNV directory not found at '{nnv_dir}'.\n"
                 f"Please check the path and ensure NNV is properly installed.")

    print(f"✅ NNV directory found at: {nnv_dir}")


    image_names = [
        '0001TP_008790.png',
        ]

    
    delta_rgb = 3

    N_perturbed_list = [17]
    
    ii=0
    for idx, image_name in enumerate(image_names):
        for N_perturbed in N_perturbed_list:
            
            print(f"Running: {image_name} with N_perturbed={N_perturbed}")
            
            
            
            Camvid_exp2( projection_batch, guarantee, device, src_dir, nnv_dir )