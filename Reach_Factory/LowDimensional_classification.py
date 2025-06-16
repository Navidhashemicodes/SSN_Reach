#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  7 11:00:00 2025

@author: hashemn
"""
import torch
import numpy as np
import onnxruntime as ort
import cv2
import time
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


class class_verifier:
    
    def __init__(self, Lb, Ub, target, nc, height, width, model, mode, device, params):
        self.Lb = Lb
        self.Ub = Ub
        self.nc = nc
        self.height = height
        self.width = width
        self.model = model
        self.mode = mode
        self.target_class = target
        self.device = device
        self.params = params
        
        
    def myFunc(self, x):
        x = x.to(torch.float16)  # Use half precision
        x_numpy = x.cpu().numpy().astype(np.float32)
        results = []
        for i in range(0, x_numpy.shape[0], self.params['sim_batch']):
            batch = x_numpy[i:i+self.params['sim_batch']]
            #with autocast():  # Automatically use mixed precision
            with torch.amp.autocast('cuda'):
                output = self.model.run(None, {self.params['input_name']: batch})
            results.append(torch.tensor(output[0]).to(self.device))
        return torch.cat(results, dim=0).T


    def Surrogate_approach(self):
        lbb = self.Lb.reshape(self.nc, self.height, self.width)
        ubb = self.Ub.reshape(self.nc, self.height, self.width)

        lbk = lbb.unsqueeze(0).repeat(self.params['Nt'], 1, 1, 1)
        ubk = ubb.unsqueeze(0).repeat(self.params['Nt'], 1, 1, 1)
        dbk = ubk - lbk

        torch.manual_seed(0)
        start_time = time.time()
        X = lbk + torch.rand_like(dbk) * dbk
        Y = self.myFunc(X)
        train_data_generation_time = time.time() - start_time

        X = X.view(self.params['Nt'], -1).T

        start_time = time.time()
        Y_centered = Y - Y.mean(dim=1, keepdim=True)
        SigmaY = (Y_centered @ Y_centered.T) / (Y.shape[1] - 1)
        eigvals, VY = torch.linalg.eigh(SigmaY)

        C = 20 * (0.001 * Y.mean(dim=1) + (0.05 - 0.001) * 0.5 * (Y.min(dim=1).values + Y.max(dim=1).values))
        dY = Y - C.unsqueeze(1)
        dYV = VY.T @ dY
        training_time = time.time() - start_time

        current_dir = os.getcwd()
        N_dir = self.params['N_dir']
        if self.mode == 'ReLU':
            save_path = os.path.join(current_dir, 'trained_relu_weights_2h_norm.mat')
            Map, Model_training_time = Trainer_ReLU(X[:,:N_dir].T, dYV[:,:N_dir].T, self.device, self.params['dims'], self.params['epochs'], save_path)
            
        if self.mode == 'Linear':
            save_path = os.path.join(current_dir, 'trained_A_b.mat')
            Map, Model_training_time = Trainer_Linear(X[:,:N_dir].T, dYV[:,:N_dir].T, self.device, self.params['epochs'], save_path)


        del dYV
        del dY
        torch.cuda.empty_cache()

        t0 = time.time()
        with torch.no_grad():
            pred = Map(X.T).T  

            approx_Y = VY @ pred + C.unsqueeze(1)  # shape: same as Y
    
        residuals = (Y - approx_Y).abs()
        threshold_normal = self.params['threshold_normal']
        res_max = residuals.max(dim=1).values
        res_max[res_max < threshold_normal] = threshold_normal

        trn_time1 = time.time()-t0


        del Y
        torch.cuda.empty_cache()
    
        start_time = time.time()

        lbk = lbb.unsqueeze(0).repeat(self.params['Ns'], 1, 1, 1)
        ubk = ubb.unsqueeze(0).repeat(self.params['Ns'], 1, 1, 1)
        dbk = ubk - lbk

        torch.manual_seed(1)
    
        X_test = lbk + torch.rand_like(dbk) * dbk
        Y_test = self.myFunc(X_test)
        test_data_generation_time = time.time() - start_time

        X_test = X_test.view(self.params['Ns'] , -1).T
        with torch.no_grad():
            pred = Map(X_test.T).T

        approx_Y = VY @ pred + C.unsqueeze(1)
    
        res_tst = (Y_test - approx_Y).abs()

    
        Rs = torch.max(torch.abs(res_tst) / res_max.unsqueeze(1), dim=0).values
        Rs_sorted, _ = torch.sort(Rs)
        R_star = Rs_sorted[self.params['rank'] - 1]  # ell is 1-based
        Conf = R_star * res_max
    
        conformal_time1 = time.time() - start_time

        current_dir = os.getcwd()
        save_path = os.path.join(current_dir, 'python_data.mat')
        c = C.cpu().numpy()
        conf = Conf.cpu().numpy()
        vy = VY.cpu().numpy()
        mylb = self.Lb.cpu().numpy()
        myub = self.Ub.cpu().numpy()

        scipy.io.savemat(save_path, {
            'Conf': conf, 'C': c, 'VY': vy, 'class': self.target_class, 'dir': self.params['src_dir'],
            'nnv_dir': self.params['nnv_dir'], 'dimp': self.width*self.height*self.nc, 'lb' : mylb , 'ub' : myub})


        del conf, c, vy


        eng = matlab.engine.start_matlab()

        if self.mode == 'ReLU':

            matlab_code = r"""

            clear
            clc
        
            load('python_data.mat')
            load('trained_relu_weights_2h_norm.mat')
            Small_net.weights = {double(W1) , double(W2), double(W3)};
            Small_net.biases = {double(b1)' , double(b2)', double(b3)'};
        
            addpath(genpath(dir))
            addpath(genpath(nnv_dir))
            
            l1 = size(W1,1);
            l2 = size(W2,1);
            l0 = size(W1,2);
            
            L = cell(l1,1);
            L(:) = {'poslin'};
            Small_net.layers{1} = L ;
            L = cell(l2,1);
            L(:) = {'poslin'};
            Small_net.layers{2} = L ;
            
            dim2 = length(Conf);
            
            tic
            H.V = [zeros(dim2,1) eye(dim2)];
            H.C = zeros(1,dim2);
            H.d = 0;
            H.predicate_lb = -Conf.';
            H.predicate_ub =  Conf.';
            H.dim = dim2;
            H.nVar= dim2;
            %%%%%%%%%%%%%%%
            conformal_time2 = toc;
            
            tic
            %%%%%%%%%%%%%%
            I = Star();
            I.V = [0.5*(lb+ub)  eye(l0)];
            I.C = zeros(1,l0);
            I.d = 0;
            I.predicate_lb = -0.5*(ub-lb);
            I.predicate_ub =  0.5*(ub-lb);
            I.dim =  l0;
            I.nVar = l0;


            Principal_reach = ReLUNN_Reachability_starinit(I, Small_net, 'approx-star');
            Surrogate_reach = affineMap(Principal_reach , VY , C.');

            %%%%%%%%%%%%%%
            reachability_time = toc;
            clear Principal_reach

            tic
            %%%%%%%%%%%%%%
            P_Center = double(Surrogate_reach.V(:,1));
            P_lb = double([Surrogate_reach.predicate_lb ; H.predicate_lb]);
            P_ub = double([Surrogate_reach.predicate_ub ; H.predicate_ub]);


            P_V = [double(Surrogate_reach.V(:,2:end))   double(H.V(:,2:end))];
            P_Center1 = P_Center(class+1,1) - P_Center;
            P_V1 = P_V(class+1, :) - P_V;
            LB = P_Center1 + 0.5*(P_V1+ abs(P_V1))*P_lb + 0.5*(P_V1 - abs(P_V1))*P_ub;


            %%%%%%%%%%%%%%
            projection_time = toc;


            clear  Surrogate_reach

            save("Matlab_data.mat", 'LB', 'projection_time', 'reachability_time', 'conformal_time2')

            """

        
        elif self.mode == 'Linear':
            
            matlab_code = r"""


            clear
            clc

            load('python_data.mat')
            load('trained_A_b.mat')
            A = double(W);
            b = double(b)';

            addpath(genpath(dir))
            addpath(genpath(nnv_dir))

            dim2 = length(Conf);

            tic
            H.V = [zeros(dim2,1) eye(dim2)];
            H.C = zeros(1,dim2);
            H.d = 0;
            H.predicate_lb = -Conf.';
            H.predicate_ub =  Conf.';
            H.dim = dim2;
            H.nVar= dim2;
            %%%%%%%%%%%%%%%
            conformal_time2 = toc;

            tic
            %%%%%%%%%%%%%%
            I = Star();
            I.V = [0.5*(lb+ub)  eye(dimp)];
            I.C = zeros(1,dimp);
            I.d = 0;
            I.predicate_lb = -0.5*(ub-lb);
            I.predicate_ub =  0.5*(ub-lb);
            I.dim =  dimp;
            I.nVar = dimp;


            Principal_reach = affineMap(I , A , b);
            Surrogate_reach = affineMap(Principal_reach , VY , C.');

            %%%%%%%%%%%%%%
            reachability_time = toc;
            clear Principal_reach

            tic
            %%%%%%%%%%%%%%
            P_Center = double(Surrogate_reach.V(:,1));
            P_lb = double([Surrogate_reach.predicate_lb ; H.predicate_lb]);
            P_ub = double([Surrogate_reach.predicate_ub ; H.predicate_ub]);


            P_V = [double(Surrogate_reach.V(:,2:end))   double(H.V(:,2:end))];
            P_Center1 = P_Center(class+1,1) - P_Center;
            P_V1 = P_V(class+1, :) - P_V;
            LB = P_Center1 + 0.5*(P_V1+ abs(P_V1))*P_lb + 0.5*(P_V1 - abs(P_V1))*P_ub;


            %%%%%%%%%%%%%%
            projection_time = toc;


            clear  Surrogate_reach

            save("Matlab_data.mat", 'LB', 'projection_time', 'reachability_time', 'conformal_time2')

            """

        eng.eval(matlab_code, nargout=0)

        eng.quit()

        current_dir = os.getcwd()
        mat_file_path = os.path.join(current_dir, 'Matlab_data.mat')
        mat_data = scipy.io.loadmat(mat_file_path)

        LB = torch.tensor(mat_data['LB'], dtype=torch.float32)
        projection_time = float(mat_data['projection_time'].item())
        reachability_time = float(mat_data['reachability_time'].item())
        conformal_time2 = float(mat_data['conformal_time2'].item())
        
        if self.mode == 'Linear':
            remove_path =  os.path.join(current_dir, 'trained_A_b.mat')
            os.remove(remove_path)
            
        if self.mode == 'ReLU':
            remove_path =  os.path.join(current_dir, 'trained_relu_weights_2h_norm.mat')
            os.remove(remove_path)
            
        
        remove_path =  os.path.join(current_dir, 'Matlab_data.mat')
        os.remove(remove_path)
        remove_path =  os.path.join(current_dir, 'python_data.mat')
        os.remove(remove_path)


        Decision = 'verified'

        if LB.min() < 0:
            Decision = 'NOT verified'

        
        Time = training_time+Model_training_time+trn_time1+conformal_time1 \
               + conformal_time2+reachability_time+projection_time
               
        print(f'Verification is complete within {Time} seconds. The final decision is: The label is {Decision}.')
        
        
        
        return Decision , Time
    
    
    
    def Naive_approach(self):
        
        
        lbb = self.Lb.reshape(self.nc, self.height, self.width)
        ubb = self.Ub.reshape(self.nc, self.height, self.width)

        lbk = lbb.unsqueeze(0).repeat(self.params['Nt'], 1, 1, 1)
        ubk = ubb.unsqueeze(0).repeat(self.params['Nt'], 1, 1, 1)
        dbk = ubk - lbk

        torch.manual_seed(0)
        start_time = time.time()
        X = lbk + torch.rand_like(dbk) * dbk
        Y = self.myFunc(X)
        train_data_generation_time = time.time() - start_time

        del X
        torch.cuda.empty_cache()

        start_time = time.time()
        Y_centered = Y - Y.mean(dim=1, keepdim=True)
        SigmaY = (Y_centered @ Y_centered.T) / (Y.shape[1] - 1)
        eigvals, VY = torch.linalg.eigh(SigmaY)

        C = 20 * (0.001 * Y.mean(dim=1) + (0.05 - 0.001) * 0.5 * (Y.min(dim=1).values + Y.max(dim=1).values))
        dY = Y - C.unsqueeze(1)
        dYV = VY.T @ dY
        r_max = torch.max(torch.abs(dYV), dim=1).values
        training_time = time.time() - start_time
        


        del Y, dYV, dY
        torch.cuda.empty_cache()

        lbk = lbb.unsqueeze(0).repeat(self.params['Ns'], 1, 1, 1)
        ubk = ubb.unsqueeze(0).repeat(self.params['Ns'], 1, 1, 1)
        dbk = ubk - lbk

        torch.manual_seed(1)
        start_time = time.time()
        X_test = lbk + torch.rand_like(dbk) * dbk
        Y_test = self.myFunc(X_test)
        test_data_generation_time = time.time() - start_time

        del X_test
        torch.cuda.empty_cache()

        start_time = time.time()
        dY_test = Y_test - C.unsqueeze(1)
        dYV_test = VY.T @ dY_test
        Rs = torch.max(torch.abs(dYV_test) / r_max.unsqueeze(1), dim=0).values
        Rs_sorted, _ = torch.sort(Rs)
        R_star = Rs_sorted[self.params['rank'] - 1]  # ell is 1-based

        d_lb = -R_star * r_max
        d_ub = R_star * r_max
        conformal_time = time.time() - start_time

        del Y_test, dYV_test, dY_test
        torch.cuda.empty_cache()

        start_time = time.time()

        C1 = C[int(self.target_class)] - C
        VY1 = VY[int(self.target_class), :] - VY

        LB = C1 + 0.5 * (VY1 + VY1.abs()) @ d_lb + 0.5 * (VY1 - VY1.abs()) @ d_ub
        verification_time = time.time() - start_time

        Decision = 'verified'

        if LB.min() < 0:
            Decision = 'NOT verified'

        Time = conformal_time + test_data_generation_time + \
               training_time + train_data_generation_time + verification_time

        print(f'Verification is complete within {Time} seconds. The final decision is: The label is {Decision}.')
        
               
        return Decision, Time