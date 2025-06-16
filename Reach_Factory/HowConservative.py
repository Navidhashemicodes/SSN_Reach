#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 14 10:48:59 2025

@author: hashemn
"""

import torch
import numpy as np
import onnxruntime as ort
import pathlib
import gc

root_dir = pathlib.Path(__file__).resolve().parents[1]



class Conservatism_analysis:
    
    
    def __init__(self, model, LB, de, LB_out, UB_out, indices, original_dim, device, params):
        
        self.de = de
        self.indices = indices
        self.device = device
        self.model = model
        self.LB = LB
        self.LB_out = LB_out
        self.UB_out = UB_out
        self.indices = indices
        self.original_dim = original_dim
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



        LBs = self.LB.repeat(repeat,1,1,1)
        Y, X = self.generate_data_chunk(repeat, LBs)


        Y = Y.view(Y.shape[0], -1)
        
        return Y
    
    
    
    def conservatism(self):
        
        Ns = self.params['Ns']
        Nsp = self.params['Nsp']
        SEED0 = self.params['SEED0']
        
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
            
            
        outsides = torch.zeros(Ns, requires_grad=False)
        ind = 0
        
        Y_max = torch.full_like(self.UB_out, -float('inf'))
        Y_min = torch.full_like(self.LB_out, float('inf'))

        for nc, curr_len in enumerate(chunk_sizes):
            
            print(f"iteration {nc} of {Num_chunks}.")
            
            Y_test = self.generate_data(curr_len, SEED0+nc+1)
            
            outsides[ind:ind + curr_len] = ((Y_test < self.LB_out) | (Y_test > (self.UB_out))).any(dim=1) 
            
            Y_max = torch.maximum(Y_max, Y_test.max(dim=0).values)
            Y_min = torch.minimum(Y_min, Y_test.min(dim=0).values)

            del Y_test
            gc.collect()
            torch.cuda.empty_cache()            
            
            ind += curr_len
        
        emprical_miscoverage = outsides.float().mean().item()
        return emprical_miscoverage, Y_min, Y_max