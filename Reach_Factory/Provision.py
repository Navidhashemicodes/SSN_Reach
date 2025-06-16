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
from scipy.stats import beta
import matlab.engine
import scipy.io
import pathlib
import gc

root_dir = pathlib.Path(__file__).resolve().parents[1]
training_factory_path = os.path.join(root_dir, 'Training_Factory')
sys.path.append(training_factory_path)

from Direction_training import compute_directions
from Training_ReLU import Trainer_ReLU
from Training_Linear import Trainer_Linear

class Reachability_provider:
    
    
    def __init__(self, model, LB, de, indices, original_dim, output_dim, device, mode, params):
        
        self.de = de
        self.indices = indices
        self.device = device
        self.model = model
        self.LB = LB
        self.original_dim = original_dim
        self.output_dim = output_dim
        self.mode = mode
        self.params = params
        
        
        
        
    def mat_generator_no_third(self, repeat, values):
        
        N_perturbed = len(self.indices)
        
        Matrix = torch.zeros( (repeat, *self.original_dim), device=values.device, dtype=values.dtype)
        
        t = 0
        for c in range(self.original_dim[0]):
            for i in range(N_perturbed):
                row, col = self.indices[i]
                Matrix[:,c,row, col] = values[:,t]
                t += 1
        return Matrix
    
    
    def Func(self, x):
        name = self.params['input_name']
        batch_size = self.params['sim_batch']
        x = x.to(torch.float16)  # Use half precision
        x_numpy = x.cpu().numpy().astype(np.float32)
        results = []
        for i in range(0, x_numpy.shape[0], batch_size):
            batch = x_numpy[i:i+batch_size]
            #with autocast():  # Automatically use mixed precision
            with torch.amp.autocast('cuda'):
                output = self.model.run(None, {name: batch})
            results.append(torch.tensor(output[0]).to(self.device))
        return torch.cat(results, dim=0)
    
    
    def generate_data_chunk(self, repeat, LBs):
        
        N_perturbed = len(self.indices)
        nc = self.original_dim[0]
        
        """ Function to generate the training data for one instance in parallel. """
        Rand = torch.rand(repeat, nc * N_perturbed).to(self.device)
        Rand_matrix = self.mat_generator_no_third(repeat, Rand)
        d_at = self.de * Rand_matrix
        Inp = LBs + d_at
        Inp_tensor = Inp.float()

        with torch.no_grad():
            out = self.Func(Inp_tensor)
    
        return out, Rand
    
    
    def generate_data(self, repeat, SEED):
        
        torch.manual_seed(SEED)

        t0 = time()


        LBs = self.LB.repeat(repeat,1,1,1)
        Y, X = self.generate_data_chunk(repeat, LBs)


        runtime = time() - t0


        Y = Y.view(Y.shape[0], -1)
        
        return Y, X, runtime
    
    
    def Shape_residual(self):
        
        Nt = self.params['Nt']
        N_dir = self.params['N_dir']
        Y_dir, X_dir, train_data_run_1_1 = self.generate_data(N_dir, 0)
        Directions, Direction_Training_time = compute_directions(Y_dir, self.device, self.params['trn_batch'])
        Directions = torch.stack([d.squeeze(-1) for d in Directions])
        

        chunck_size = N_dir
        Num_chunks = Nt // chunck_size
        remainder = Nt % chunck_size

        chunk_sizes = [chunck_size] * Num_chunks
        if remainder != 0:
            chunk_sizes.append(remainder)

        train_data_run_1_l = []
        train_data_run_1_l.append(train_data_run_1_1)
        
        C_l = []
        YV_l = []
        X_l =[]
        Y_l = []
        
        X_dir = X_dir.cpu()
        X_l.append(X_dir)
        C =  20 * (0.001 * Y_dir.mean(dim=0) + (0.05 - 0.001) * 0.5 * (Y_dir.min(dim=0).values + Y_dir.max(dim=0).values))
        C_l.append(C)
        YV_l.append(Y_dir @ Directions.T)
        Y_dir = Y_dir.cpu()
        Y_l.append(Y_dir)
        gc.collect()
        torch.cuda.empty_cache()
        for nc, curr_len in enumerate(chunk_sizes[1:], start=1):
            Y_dir, X_dir, tYX = self.generate_data(curr_len, nc)
            train_data_run_1_l.append(tYX)
            X_dir = X_dir.cpu()
            X_l.append(X_dir)
            C =  20 * (0.001 * Y_dir.mean(dim=0) + (0.05 - 0.001) * 0.5 * (Y_dir.min(dim=0).values + Y_dir.max(dim=0).values))
            C_l.append(C)
            YV_l.append(Y_dir @ Directions.T)
            Y_dir = Y_dir.cpu()
            Y_l.append(Y_dir)
            gc.collect()
            torch.cuda.empty_cache()
            
            
        stackedC = torch.stack(C_l, dim=0)
        C = stackedC.mean(dim=0)
        CV = C @ Directions.T
        YV = torch.cat(YV_l , dim=0)
        dYV = YV - CV.unsqueeze(0)
        X = torch.cat(X_l , dim=0).to(self.device)
        train_data_run_1 = sum(train_data_run_1_l)
        del YV, CV, C_l, YV_l
        gc.collect()
        torch.cuda.empty_cache()
        
        current_dir = os.getcwd()
        if self.mode == 'ReLU':
            save_path = os.path.join(current_dir, 'trained_relu_weights_2h_norm.mat')
            Map, Model_training_time = Trainer_ReLU(X, dYV, self.device, self.params['dims'], self.params['epochs'], save_path)
            
        if self.mode == 'Linear':
            save_path = os.path.join(current_dir, 'trained_A_b.mat')
            Map, Model_training_time = Trainer_Linear(X, dYV, self.device, self.params['epochs'], save_path)
        
        trn_time1_l = []
        res_max_l = []

        for nc, curr_len in enumerate(chunk_sizes):
            Y = Y_l[nc].to(self.device) 
            X = X_l[nc].to(self.device)
            
            t0 = time()
            with torch.no_grad():
                pred = Map(X) 
                approx_Y = pred @ Directions  + C.unsqueeze(0)  # shape: same as Y

            residuals = (Y - approx_Y).abs()
            res_max_l.append( residuals.max(dim=0).values)
            trn_time1_l.append( time() - t0)
            del Y, approx_Y, pred, residuals
            gc.collect()
            torch.cuda.empty_cache()
        
        del Y_l, X_l
        t0 = time()
        stacked_max = torch.stack(res_max_l, dim=0)
        res_max = stacked_max.max(dim=0).values
        tn = self.params['threshold_normal']
        res_max[res_max < tn ] = tn
        trn_time1_l.append( time()-t0 )
        trn_time1 = sum(trn_time1_l)
        
        return res_max, C, Directions, Map, trn_time1, train_data_run_1, Direction_Training_time, Model_training_time

    
    def CI_surrogate(self, Map, C, res_max, Directions):
        
        Ns = self.params['Ns']
        Nsp = self.params['Nsp']
        Nt = self.params['Nt']
        N_dir = self.params['N_dir']
        
        if Nt % N_dir == 0:
            seed_loc = Nt // N_dir
        else:
            seed_loc = 1 +  ( Nt // N_dir )

        ell = self.params['rank']
        
        
        thelen = min(Ns, Nsp)
        if Ns > thelen:
            chunck_size = thelen
            Num_chunks = Ns // chunck_size
            remainder = Ns % chunck_size
        else:
            chunck_size = Ns
            Num_chunks = 1
            remainder = 0

        chunk_sizes = [chunck_size] * Num_chunks
        if remainder != 0:
            chunk_sizes.append(remainder)

        Rs = torch.zeros(Ns, requires_grad=False)
        ind = 0
        test_data_run = []
        res_test_time = []



        for nc, curr_len in enumerate(chunk_sizes):
            
            Y_test, X_test_nc, tst_run = self.generate_data(curr_len, seed_loc+nc+1)
            test_data_run.append( tst_run )
            
            t1 = time()
            with torch.no_grad():
                pred = Map(X_test_nc) @ Directions  + C.unsqueeze(0)  # shape: (dim2, curr_len)
            res_tst = (Y_test - pred).abs()
            vals = torch.max(res_tst / res_max.unsqueeze(0), dim=1).values
            res_test_time.append(time() - t1)
            del Y_test, pred, res_tst
            gc.collect()
            torch.cuda.empty_cache()
            Rs[ind:ind + curr_len] = vals 

            ind += curr_len
        
        t0 = time()

        with torch.no_grad():
            Rs_sorted = torch.sort(Rs).values
            R_star = Rs_sorted[ell-1]  # Assuming `ell` is defined
            Conf = R_star * res_max

        conformal_time = time() - t0
            
        return Conf, R_star, conformal_time, res_test_time, test_data_run
      
    
    def Provider(self):
        
        
        res_max, C, Directions, Map, trn_time1, train_data_run_1, Direction_Training_time, Model_training_time = self.Shape_residual()
        
        Conf, R_star, conformal_time, res_test_time, test_data_run = self.CI_surrogate(Map, C, res_max, Directions)
        
        
        save_dict = {
            "Conf": Conf,
            "Directions" : Directions,
            "C" : C,
            "Map" : Map,
            "R_star": R_star,
            "res_max": res_max,
            "train_data_run_1": train_data_run_1,
            "trn_time1": trn_time1,
            "test_data_run": test_data_run,
            "res_test_time": res_test_time,
            "conformal_time": conformal_time,
            "Direction_Training_time": Direction_Training_time,
            "Model_training_time": Model_training_time,
            "Ns" : self.params['Ns'],
            "Nsp" : self.params['Nsp'],
            "rank" : self.params['rank'],
            "Nt" : self.params['Nt'],
            "N_dir" : self.params['N_dir'],
            "threshold_normal" : self.params['threshold_normal'],
            "perturbation" : self.params['perturbation'],
            "True_class" : self.params['True_class'],
            "class_threshold" : self.params['class_threshold'],
            "image_name" : self.params['image_name'],
            "de" : self.de,
            "indices" : self.indices,
            "original_dim" : self.original_dim,
            "output_dim" : self.output_dim,
            "mode" : self.mode,
            "trn_batch" : self.params['trn_batch'],
            "sim_batch" : self.params['sim_batch'],
            "dims" : self.params['dims'],
            "epochs" : self.params['epochs']
            }

        
        for key, val in save_dict.items():
            if isinstance(val, torch.Tensor):
                save_dict[key] = val.cpu()
            elif isinstance(val, list):
                save_dict[key] = [v.cpu() if isinstance(v, torch.Tensor) else v for v in val]
            elif isinstance(val, dict):
                save_dict[key] = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in val.items()}
        
        
        save_name = 'CI_provider.pt'
        torch.save(save_dict, save_name)
    
        
    
    
    
    
    
    