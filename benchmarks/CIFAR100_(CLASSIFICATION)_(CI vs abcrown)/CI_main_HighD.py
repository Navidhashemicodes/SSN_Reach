#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  7 03:24:42 2025

@author: hashemn
"""


import torch
import numpy as np
import onnxruntime as ort
import os
import sys
import pathlib


sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
root_dir = pathlib.Path(__file__).resolve().parents[2]
reach_factory_path = os.path.join(root_dir, 'Reach_Factory')
sys.path.append(reach_factory_path)
from Provision import Reachability_provider
from Segmentation import Segmentor
import os
import scipy.io
import onnx





def CIFAR100_1( label, Lb, de, k, Nt, N_dir,
            Ns, Nsp, rank, device, threshold_normal,
            sim_batch, trn_batch, epochs, surrogate_mode, dims):
    
    model_path = 'CIFAR100_resnet_large.onnx'
    model = onnx.load(model_path)
    name = model.graph.input[0].name
    ort_session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
    
    
    output_dim = [1, 100, 1, 1]
    True_class = [[int(label) ]]

    indices = [[i,j] for i in range(32) for j in range(32)]        
    indices = np.array(indices)
    


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
        'perturbation' : k,
        'True_class' : True_class,
        'class_threshold' : None,
        'image_name' : f'(Bounds_{k})',
        'input_name': name
    }
    
    
    provide = Reachability_provider(
        de = de,
        indices = indices,
        device = device,
        model = ort_session,
        LB = Lb,
        original_dim = (3, 32, 32),
        output_dim = output_dim,
        mode = surrogate_mode,
        params = params
        )
    
    provide.Provider()




def CIFAR100_2( projection_batch, guarantee, device, src_dir, nnv_dir ):
    
    params = {
        'guarantee': guarantee,
    }
    
    myclass = Segmentor(
        device = device,
        src_dir = src_dir,
        nnv_dir = nnv_dir,
        params = params
        )

    Decision, Time = myclass.is_classified()
    
    return Decision, Time




if __name__ == '__main__':
    
    K = 20
    Nt = 2000
    Ns = 100000
    Nsp = Ns
    rank = 99999
    epochs = 40;
    guarantee = 0.9999
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    N_dir = Nt
    threshold_normal = 1e-5
    sim_batch = 500
    trn_batch = 20
    surrogate_mode = 'ReLU'
    src_dir = os.path.join(root_dir, 'src')
    nnv_dir = '/home/hashemn/nnv'
    dims = [ 100 , 'auto' ]
    projection_batch = 100
    
    if not os.path.isdir(nnv_dir):
        sys.exit(f"❌ Error: NNV directory not found at '{nnv_dir}'.\n"
                 f"Please check the path and ensure NNV is properly installed.")

    print(f"✅ NNV directory found at: {nnv_dir}")


    
    Runtimes = []
    Results = []
    for i in range(20):
        
        mat = scipy.io.loadmat(f'Bounds/Bounds_{i+1}.mat')  # Replace with your actual .mat filename

        lower = mat['lb']
        upper = mat['ub']
        label = mat['label'].squeeze()
        lb0 = torch.tensor(lower, dtype=torch.float32, device=device)
        ub0 = torch.tensor(upper, dtype=torch.float32, device=device)

        Times = []
        Decisions = []

        for k in range(1, K + 1):
            
            mid = 0.5 * (ub0 + lb0)
            half_range = 0.5 * (ub0 - lb0)

            lbb = (mid - k * half_range).reshape(3, 32, 32)
            ubb = (mid + k * half_range).reshape(3, 32, 32)

            lbk = lbb.unsqueeze(0)
            ubk = ubb.unsqueeze(0)
            dbk = ubk - lbk
            
            CIFAR100_1( label, lbk, dbk, k, Nt, N_dir,
                        Ns, Nsp, rank, device, threshold_normal,
                        sim_batch, trn_batch, epochs, surrogate_mode, dims)
            
            
            Decision, Time = CIFAR100_2( projection_batch, guarantee, device, src_dir, nnv_dir )
            
            Times.append(Time)
            Decisions.append(Decision)
        
        Results.append(Decisions)
        Runtimes.append(Times)