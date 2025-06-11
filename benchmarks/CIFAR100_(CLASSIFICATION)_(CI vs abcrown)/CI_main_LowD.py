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
from scipy.stats import beta


sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
root_dir = pathlib.Path(__file__).resolve().parents[2]
reach_factory_path = os.path.join(root_dir, 'Reach_Factory')
sys.path.append(reach_factory_path)
from LowDimensional_classification import class_verifier
import os
import scipy.io
import onnx



def CIFAR100( Lb, Ub, label, nc, height, width, src_dir, nnv_dir, device, N_dir,
             Nt, Ns, rank, epochs, sim_batch, dims, threshold_normal, surrogate_mode ):
    
    model_path = 'CIFAR100_resnet_large.onnx'
    model = onnx.load(model_path)
    name = model.graph.input[0].name
    ort_session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
    


    params = {
        'sim_batch' : sim_batch,
        'Nt' : Nt,
        'N_dir' : N_dir,
        'dims' : dims,
        'epochs' : epochs,
        'threshold_normal' : threshold_normal,
        'Ns' : Ns,
        'rank' : rank,
        'input_name': name,
        'nnv_dir' : nnv_dir,
        'src_dir' : src_dir
    }
    
    
    verify = class_verifier(
        Lb = Lb,
        Ub = Ub,
        target = label,
        nc = nc,
        height = height,
        width = width,
        model = ort_session,
        mode = surrogate_mode,
        device = device,
        params = params
        )
    
    Decision, Time = verify.Surrogate_approach()
    
    return Decision, Time

def CIFAR100_Naive( Lb, Ub, label, nc, height, width, device,
             Nt, Ns, rank, sim_batch, threshold_normal ):
    
    model_path = 'CIFAR100_resnet_large.onnx'
    model = onnx.load(model_path)
    name = model.graph.input[0].name
    ort_session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
    


    params = {
        'sim_batch' : sim_batch,
        'Nt' : Nt,
        'threshold_normal' : threshold_normal,
        'Ns' : Ns,
        'rank' : rank,
        'input_name': name
    }
    
    
    verify = class_verifier(
        Lb = Lb,
        Ub = Ub,
        target = label,
        nc = nc,
        height = height,
        width = width,
        model = ort_session,
        mode = None,
        device = device,
        params = params
        )
    
    Decision, Time = verify.Naive_approach()
    
    return Decision, Time



if __name__ == '__main__':
    
    surrogate_mode = 'ReLU'
    
    
    if surrogate_mode == 'Linear':
        
        K = 30
        Nt = 2000
        N_dir = Nt
        Ns = 100000
        rank = 99999
        epochs = 200;
        guarantee = 0.9999
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        threshold_normal = 1e-5
        sim_batch = 500
        trn_batch = 20
        src_dir = os.path.join(root_dir, 'src')
        nnv_dir = '/home/hashemn/nnv'
        dims = None
        
        if not os.path.isdir(nnv_dir):
            sys.exit(f"❌ Error: NNV directory not found at '{nnv_dir}'.\n"
                     f"Please check the path and ensure NNV is properly installed.")

        print(f"✅ NNV directory found at: {nnv_dir}")

        Failure_chance_of_guarantee = beta.cdf(guarantee, rank, Ns + 1 - rank)
        print( f'the guarantee for the verification is Pr[ Pr[ Label_Remains_Valid] > {guarantee} ] > {1-Failure_chance_of_guarantee}')

        Runtimes = []
        Results = []
        for i in range(200):
            
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
                
                Lb = mid - k * half_range
                Ub = mid + k * half_range
                nc = 3
                height = 32
                width = 32 
                
                
                Decision, Time = CIFAR100( Lb, Ub, label, nc, height, width, src_dir, nnv_dir, device, N_dir,
                                          Nt, Ns, rank, epochs, sim_batch, dims, threshold_normal, surrogate_mode )
                
                
                if Decision == 'verified':
                    Times.append(Time)
                    Decisions.append(Decision)
                else:
                    break
            
            Results.append(Decisions)
            Runtimes.append(Times)
            
            torch.save({'Results': Results, 'Runtimes': Runtimes}, 'From_lowD_class_LinearSurrogate.pt')
        
    if surrogate_mode == 'ReLU':
        
        K = 30
        Nt = 2000
        N_dir = Nt
        Ns = 100000
        rank = 99999
        epochs = 60;
        guarantee = 0.9999
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        threshold_normal = 1e-5
        sim_batch = 500
        trn_batch = 20

        src_dir = os.path.join(root_dir, 'src')
        nnv_dir = '/home/hashemn/nnv'
        dims = [ 100 , 'auto' ]
        
        if not os.path.isdir(nnv_dir):
            sys.exit(f"❌ Error: NNV directory not found at '{nnv_dir}'.\n"
                     f"Please check the path and ensure NNV is properly installed.")

        print(f"✅ NNV directory found at: {nnv_dir}")

        Failure_chance_of_guarantee = beta.cdf(guarantee, rank, Ns + 1 - rank)
        print( f'the guarantee for the verification is Pr[ Pr[ Label_Remains_Valid] > {guarantee} ] > {1-Failure_chance_of_guarantee}')

        Runtimes = []
        Results = []
        for i in range(200):
            
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
                
                Lb = mid - k * half_range
                Ub = mid + k * half_range
                nc = 3
                height = 32
                width = 32 
                
                
                Decision, Time = CIFAR100( Lb, Ub, label, nc, height, width, src_dir, nnv_dir, device, N_dir,
                                          Nt, Ns, rank, epochs, sim_batch, dims, threshold_normal, surrogate_mode )
                
                
                
                if Decision == 'verified':
                    Times.append(Time)
                    Decisions.append(Decision)
                else:
                    break
            
            Results.append(Decisions)
            Runtimes.append(Times)
            torch.save({'Results': Results, 'Runtimes': Runtimes}, 'From_lowD_class_ReLUSurrogate.pt')
        
    if surrogate_mode == 'Naive':
        
        K = 10
        Nt = 2000
        Ns = 100000
        rank = 99999
        guarantee = 0.9999
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        threshold_normal = 1e-5
        sim_batch = 500


        Failure_chance_of_guarantee = beta.cdf(guarantee, rank, Ns + 1 - rank)
        # print( f'the guarantee for the verification is Pr[ Pr[ Label_Remains_Valid] > {guarantee} ] > {1-Failure_chance_of_guarantee}')

        Runtimes = []
        Results = []
        for i in range(200):
            
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
                
                Lb = mid - k * half_range
                Ub = mid + k * half_range
                nc = 3
                height = 32
                width = 32 
                
                
                Decision, Time = CIFAR100_Naive( Lb, Ub, label, nc, height, width,device,
                                                 Nt, Ns, rank, sim_batch, threshold_normal )
                
                
                
                if Decision == 'verified':
                    Times.append(Time)
                    Decisions.append(Decision)
                else:
                    break
            
            Results.append(Decisions)
            Runtimes.append(Times)
            torch.save({'Results': Results, 'Runtimes': Runtimes}, 'From_lowD_class_NoSurrogate.pt')
        
        
    
    