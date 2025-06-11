# -*- coding: utf-8 -*-
"""
Created on Wed May 21 23:57:52 2025

@author: navid
"""
import torch
import numpy as np
from time import time
import os
import sys
from scipy.stats import beta
import matlab.engine
import scipy.io
import pathlib



class classification:
    
    
    def __init__(self, device, src_dir, nnv_dir, params):
        
        self.device = device
        self.src_dir = src_dir
        self.nnv_dir = nnv_dir
        self.params = params



    def Reach_ReLU(self):
        
        
        current_dir = os.getcwd()
        save_path = os.path.join(current_dir, 'directories.mat')
        scipy.io.savemat(save_path, {'dir': self.src_dir, 'nnv_dir': self.nnv_dir} )
        
        
        eng = matlab.engine.start_matlab()

        matlab_code = r"""


            clear
            clc
            
            load('directories.mat')
            load('trained_relu_weights_2h_norm.mat')
            Small_net.weights = {double(W1) , double(W2), double(W3)};
            Small_net.biases = {double(b1)' , double(b2)', double(b3)'};
                                
            l1 = size(W1,1);
            l2 = size(W2,1);
            l0 = size(W1,2);
            
            L = cell(l1,1);
            L(:) = {'poslin'};
            Small_net.layers{1} = L ;
            L = cell(l2,1);
            L(:) = {'poslin'};
            Small_net.layers{2} = L ;
            
            addpath(genpath(dir))
            addpath(genpath(nnv_dir))

            tic
            %%%%%%%%%%%%%%
            I = Star();
            I.V = [0.5*ones(l0,1)  eye(l0)];
            I.C = zeros(1,l0);
            I.d = 0;
            I.predicate_lb = -0.5*ones(l0,1);
            I.predicate_ub =  0.5*ones(l0,1);
            I.dim =  l0;
            I.nVar = l0;


            Principal_reach = ReLUNN_Reachability_starinit(I, Small_net, 'approx-star');
            
            center = Principal_reach.V(:,1);
            generators = Principal_reach.V(:, 2:end);
            pred_lb = Principal_reach.predicate_lb;
            pred_ub = Principal_reach.predicate_ub;
            
            reachability_time = toc

            save("Matlab_data.mat", 'center', 'generators', 'pred_lb', 'pred_ub', 'reachability_time')

            """
            
        eng.eval(matlab_code, nargout=0)
        eng.quit()
        
        
        current_dir = os.getcwd()
        mat_file_path = os.path.join(current_dir, 'Matlab_data.mat')
        mat_data = scipy.io.loadmat(mat_file_path)

        center = torch.tensor(mat_data['center'], dtype=torch.float32, device= self.device)
        generators = torch.tensor(mat_data['generators'], dtype=torch.float32, device= self.device)
        pred_lb = torch.tensor(mat_data['pred_lb'], dtype=torch.float32, device= self.device)
        pred_ub = torch.tensor(mat_data['pred_ub'], dtype=torch.float32, device= self.device)
        reachability_time = float(mat_data['reachability_time'].item())
        
        print('Surrogate Reachability is Done!!')
        remove_path =  os.path.join(current_dir, 'Matlab_data.mat')
        os.remove(remove_path)
        
        remove_path =  os.path.join(current_dir, 'directories.mat')
        os.remove(remove_path)
        
        remove_path =  os.path.join(current_dir, 'trained_relu_weights_2h_norm.mat')
        os.remove(remove_path)
        
        
        return center, generators , pred_lb, pred_ub, reachability_time
        
    def Reach_Linear(self, Map):
        
        reach_start = time()
        A = Map.linear.weight.detach()
        b = Map.linear.bias.detach()
        reach_start = time()
        l0 = A.shape[1]
        I_center = 0.5 * torch.ones((l0,), device= self.device)
        I_lb = -0.5 * torch.ones((l0,), device= self.device)
        I_ub =  0.5 * torch.ones((l0,), device= self.device)    
        center = (A @ I_center + b).unsqueeze(1)
        generators = A
        pred_lb = I_lb
        pred_ub = I_ub
        reachability_time = time() - reach_start
        
        print('Surrogate Reachability is Done!!')
        
        current_dir = os.getcwd()
        remove_path =  os.path.join(current_dir, 'trained_A_b.mat')
        os.remove(remove_path)
        
        return center, generators , pred_lb, pred_ub, reachability_time
        
        
    def Verify_with_surrogate( self, Map, Conf, Directions, C, dimp, mode, output_dim ):
        

        dim2 = Conf.shape[0]
       
            
        H_lb = -Conf
        H_ub = Conf
            
        if mode == 'ReLU':
            center, generators , pred_lb, pred_ub, reach_time = self.Reach_ReLU()
            pred_ub = pred_ub.squeeze()
            pred_lb = pred_lb.squeeze()
            
        if mode == 'Linear':
            center, generators , pred_lb, pred_ub, reach_time = self.Reach_Linear(Map)
            
            
        S_center = Directions.T @ center + C.unsqueeze(1)  ###????????
            
        proj_start = time()
        batch_size = self.params['projection_batch']
        num_chunks = dim2 // batch_size + (dim2 % batch_size > 0)
        Lb_list, Ub_list = [], []
            
        for i in range(num_chunks):
            start = i * batch_size
            end = min((i + 1) * batch_size, dim2)
                
            SR = Directions[:, start:end].T @ generators
            P_c = S_center[start:end]
                
            SR_pos = 0.5 * (SR + SR.abs())
            SR_neg = 0.5 * (SR - SR.abs())
            del SR


            HL = H_lb[start:end].unsqueeze(1)
            HU = H_ub[start:end].unsqueeze(1)
                
            Lb = P_c + SR_pos @ pred_lb.unsqueeze(1) +  HL + SR_neg @ pred_ub.unsqueeze(1) 
            Ub = P_c + SR_pos @ pred_ub.unsqueeze(1) +  HU + SR_neg @ pred_lb.unsqueeze(1) 
            del SR_pos, SR_neg                    
            Lb_list.append(Lb)
            Ub_list.append(Ub)
            del Lb, Ub
            
        Lb = torch.cat(Lb_list).squeeze().cpu()
        Ub = torch.cat(Ub_list).squeeze().cpu()
        projection_time = time() - proj_start

        # Reshape to 2D images
        Lb_pixels = Lb.view(dimp, -1).numpy()
        Ub_pixels = Ub.view(dimp, -1).numpy()
        
        n_batch = output_dim[0]
        n_class = output_dim[1]
        height = output_dim[2]
        width = output_dim[3]
        logits = Lb.reshape(n_batch, n_class, height, width)   
        logits = logits.permute(0, 2, 3, 1)            
        Lb_pixels = logits.reshape(1, -1, n_class).squeeze(0)
        logits = Ub.reshape(n_batch, n_class, height, width)            
        logits = logits.permute(0, 2, 3, 1)           
        Ub_pixels = logits.reshape(1, -1, n_class).squeeze(0)

        return Lb_pixels, Ub_pixels, projection_time, reach_time


    def Mask_titles(self):
        
        current_dir = os.getcwd()
        file_path = os.path.join(current_dir, 'CI_provider.pt')
        Data = torch.load(file_path, weights_only=False)
    
        N_perturbed = len(Data['indices']) 
        Ns = Data['Ns']
        guarantee = self.params['guarantee']
        ell = Data['rank']
        Failure_chance_of_guarantee = beta.cdf(guarantee, ell, Ns + 1 - ell)

        Map = Data['Map'].to(self.device)
        Conf = Data['Conf'].to(self.device)
        Directions = Data['Directions'].to(self.device)
        C = Data['C'].to(self.device)
        
        dimp = Data['original_dim'][1]*Data['original_dim'][2]
        mode = Data['mode']
        output_dim = Data['output_dim']
        print('Reachability started...')
        Lb_pixels, Ub_pixels, projection_time, reachability_time = self.Verify_with_surrogate(Map , Conf, Directions, C, dimp, mode, output_dim)
        print('Reachability is finished and projection is done!!')
    
    
        start_time = time()
    
        height = Data['output_dim'][2]
        width= Data['output_dim'][3]
        mask_dim = Data['output_dim'][1]
    
        if mask_dim == 1:
        
        # Some SSNs with two classes have one dimensional logits
        # where each class is found using a threshold on this logit
        
            classes = [[None for _ in range(height)] for _ in range(width)]
            class_threshold = Data['class_threshold']
            for i in range(height):
                for j in range(width):
                    t = i * width + j  
                    lb = Lb_pixels[t].item()
                    ub = Ub_pixels[t].item()
                
                    if lb > class_threshold:
                        class_members = [1]
                    elif ub <= class_threshold:
                        class_members = [0]
                    else:
                        class_members = [0, 1]
                    classes[i][j] = class_members
                
        else:
            
        
            Lb_max, _ = torch.max(Lb_pixels, dim=1, keepdim=True)  # Shape: [720*960, 1]

            mask = Lb_max <= Ub_pixels 

            mask_np = mask.cpu().numpy()

            classes = [[[] for _ in range(width)] for _ in range(height)]

            for t in range(width * height):
                j = t % width
                i = t // width
                class_members = [k for k in range(mask_dim) if mask_np[t, k]]
                classes[i][j] = class_members

        
        
        # Initialize counters
        robust = 0
        nonrobust = 0
        unknown = 0
        attacked = N_perturbed  # Assuming this is defined
        True_class = Data['True_class']

        for i in range(height):
            for j in range(width):
                if len(classes[i][j]) == 1:
                    if classes[i][j] == [True_class[i][j]]:
                        robust += 1
                    else:
                        nonrobust += 1
                else:
                    if True_class[i][j] in classes[i][j]:
                        unknown += 1
                    else:
                        nonrobust += 1
        
        
        Pixel_status_time = time() - start_time
        
        
        # Compute the robustness percentage
        dim_pic = height*width
        RV = 100 * robust / dim_pic
    
        print(f"Number of Robust pixels: {robust}")
        print(f"Number of non-Robust pixels: {nonrobust}")
        print(f"Number of unknown pixels: {unknown}")
        print(f"RV value: {RV}")
    

    
        print(f"Pr[ Pr[ RV value = {RV} ] > {guarantee}%] > {1 - Failure_chance_of_guarantee}")


        verification_runtime = Data['train_data_run_1'] + Data['trn_time1'] + sum(Data['test_data_run']) + \
                               sum(Data['res_test_time']) + Data['conformal_time'] + reachability_time + \
                               projection_time + Data['Direction_Training_time'] + Data['Model_training_time'] + \
                               Pixel_status_time

        print(f"The verification runtime is: {verification_runtime / 60:.2f} minutes.")


        save_dict = {
        "robust": robust,
        "nonrobust": nonrobust,
        "attacked": attacked,
        "unknown": unknown,
        "True_class": True_class,
        "classes": classes,
        "class_threshold" : Data['class_threshold'],
        "image_name" : Data['image_name'],
        "Conf": Conf,
        "Nt": Data['Nt'],
        "N_dir": Data['N_dir'],
        "de": Data['de'],
        "ell": ell,
        "Lb_pixels": Lb_pixels,
        "Ub_pixels": Ub_pixels,
        "Ns": Ns,
        "Nsp": Data['Nsp'],
        "R_star": Data['R_star'],
        "res_max": Data['res_max'],
        "RV": RV,
        "guarantee" : guarantee,
        "verification_runtime": verification_runtime,
        "threshold_normal": Data['threshold_normal'],
        "train_data_run_1": Data['train_data_run_1'],
        "trn_time1": Data['trn_time1'],
        "test_data_run": Data['test_data_run'],
        "res_test_time": Data['res_test_time'],
        "conformal_time": Data['conformal_time'],
        "reachability_time": reachability_time,
        "projection_time": projection_time,
        "Direction_Training_time": Data['Direction_Training_time'],
        "Model_training_time": Data['Model_training_time'],
        "Pixel_status_time" : Pixel_status_time,
        "projection_batch" : self.params['projection_batch'],
        "trn_batch" : Data['trn_batch'],
        "sim_batch" : Data['sim_batch'],
        "dims" : Data['dims'],
        "epochs" : Data['epochs'],
        "perturbation" : Data['perturbation'],
        "device" : self.device,
        "mode" : mode,
        "original_dim" : Data['original_dim'],
        "output_dim" : output_dim
        }


        for key, val in save_dict.items():
            if isinstance(val, torch.Tensor):
                save_dict[key] = val.cpu()
            elif isinstance(val, list):
                save_dict[key] = [v.cpu() if isinstance(v, torch.Tensor) else v for v in val]
            elif isinstance(val, dict):
                save_dict[key] = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in val.items()}
    
        # save_name = f"CI_result_middle_guarantee_ReLU_relaxed_eps_{delta_rgb}_Npertubed_{N_perturbed}"+image_name+".pt"
        base_name = os.path.splitext(Data['image_name'])[0]
    
        if mode == 'ReLU':
            save_name = f"CI_result_ReLU_relaxed_eps_{Data['perturbation']}_Npertubed_{N_perturbed}_{base_name}.pt"
    
        if mode == 'Linear':
            save_name = f"CI_result_Linear_relaxed_eps_{Data['perturbation']}_Npertubed_{N_perturbed}_{base_name}.pt"
        torch.save(save_dict, save_name)
        os.remove(file_path)
        
        print('All the details are saved')