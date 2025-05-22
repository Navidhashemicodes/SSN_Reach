import torch
import numpy as np
import onnxruntime as ort
import time
from torch.cuda.amp import autocast
import os
import matlab.engine
import scipy.io

def CI(K, Nt, epochs, Nc, ell, target_class, lower, upper):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epsilon = 0.001

    # Load ONNX model with GPU provider
    model_path = 'CIFAR100_resnet_large.onnx'
    ort_session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])


    def CIFAR100Func(x, batch_size=500):
        x = x.to(torch.float16)  # Use half precision
        x_numpy = x.cpu().numpy().astype(np.float32)
        results = []
        for i in range(0, x_numpy.shape[0], batch_size):
            batch = x_numpy[i:i+batch_size]
            #with autocast():  # Automatically use mixed precision
            with torch.amp.autocast('cuda'):
                output = ort_session.run(None, {'modelInput': batch})
            results.append(torch.tensor(output[0]).to(device))
        return torch.cat(results, dim=0).T

    lb0 = torch.tensor(lower, dtype=torch.float32, device=device)
    ub0 = torch.tensor(upper, dtype=torch.float32, device=device)

    Times = []
    Decisions = []

    for k in range(1, K + 1):
        mid = 0.5 * (ub0 + lb0)
        half_range = 0.5 * (ub0 - lb0)

        lbb = (mid - k * half_range).reshape(3, 32, 32)
        ubb = (mid + k * half_range).reshape(3, 32, 32)

        lbk = lbb.unsqueeze(0).repeat(Nt, 1, 1, 1)
        ubk = ubb.unsqueeze(0).repeat(Nt, 1, 1, 1)
        dbk = ubk - lbk

        torch.manual_seed(0)
        start_time = time.time()
        X = lbk + torch.rand_like(dbk) * dbk
        Y = CIFAR100Func(X)
        train_data_generation_time = time.time() - start_time

        X = X.view(Nt, -1).T

        start_time = time.time()
        Y_centered = Y - Y.mean(dim=1, keepdim=True)
        SigmaY = (Y_centered @ Y_centered.T) / (Y.shape[1] - 1)
        eigvals, VY = torch.linalg.eigh(SigmaY)

        C = 20 * (epsilon * Y.mean(dim=1) + (0.05 - epsilon) * 0.5 * (Y.min(dim=1).values + Y.max(dim=1).values))
        dY = Y - C.unsqueeze(1)
        dYV = VY.T @ dY
        training_time = time.time() - start_time


        from Training_ReLU import Trainer_ReLU
        current_dir = os.getcwd()
        save_path = os.path.join(current_dir, 'trained_relu_weights_2h_norm.mat')

        Map, Model_training_time = Trainer_ReLU(X, dYV, device, epochs, save_path)

        del dYV
        del dY
        torch.cuda.empty_cache()

        t0 = time.time()
        with torch.no_grad():
            pred = Map(X.T).T  

            approx_Y = VY @ pred + C.unsqueeze(1)  # shape: same as Y
        
        residuals = (Y - approx_Y).abs()
        threshold_normal = 1e-5
        res_max = residuals.max(dim=1).values
        res_max[res_max < threshold_normal] = threshold_normal

        trn_time1 = time.time()-t0


        del Y
        torch.cuda.empty_cache()
        
        start_time = time.time()

        lbk = lbb.unsqueeze(0).repeat(Nc, 1, 1, 1)
        ubk = ubb.unsqueeze(0).repeat(Nc, 1, 1, 1)
        dbk = ubk - lbk

        torch.manual_seed(1)
        
        X_test = lbk + torch.rand_like(dbk) * dbk
        Y_test = CIFAR100Func(X_test)
        test_data_generation_time = time.time() - start_time

        X_test = X_test.view(Nc , -1).T
        with torch.no_grad():
            pred = Map(X_test.T).T  

            approx_Y = VY @ pred + C.unsqueeze(1)
        
        res_tst = (Y_test - approx_Y).abs()

        
        Rs = torch.max(torch.abs(res_tst) / res_max.unsqueeze(1), dim=0).values
        Rs_sorted, _ = torch.sort(Rs)
        R_star = Rs_sorted[ell - 1]  # ell is 1-based
        Conf = R_star * res_max
        
        conformal_time1 = time.time() - start_time

        current_dir = os.getcwd()
        save_path = os.path.join(current_dir, 'python_data.mat')
        c = C.cpu().numpy()
        conf = Conf.cpu().numpy()
        vy = VY.cpu().numpy()

        scipy.io.savemat(save_path, {
            'Conf': conf, 'C': c, 'VY': vy, 'class': target_class})

        del conf, c, vy


        eng = matlab.engine.start_matlab()


        matlab_code = r"""


        clear
        clc

        load('python_data.mat')
        load('trained_relu_weights_2h_norm.mat')
        Small_net.weights = {double(W1) , double(W2), double(W3)};
        Small_net.biases = {double(b1)' , double(b2)', double(b3)'};
        load('Bounds.mat')

        l1 = size(W1,1);
        l2 = size(W2,1);
        l0 = size(W1,2);

        L = cell(l1,1);
        L(:) = {'poslin'};
        Small_net.layers{1} = L ;
        L = cell(l2,1);
        L(:) = {'poslin'};
        Small_net.layers{2} = L ;

        dim2 = 100
        dimp = 3072

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

        addpath(genpath('Path_to\src'))
        addpath(genpath('Path_to_\nnv'))

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

        eng.eval(matlab_code, nargout=0)

        eng.quit()

        current_dir = os.getcwd()
        mat_file_path = os.path.join(current_dir, 'Matlab_data.mat')
        mat_data = scipy.io.loadmat(mat_file_path)

        LB = torch.tensor(mat_data['LB'], dtype=torch.float32)
        projection_time = float(mat_data['projection_time'].item())
        reachability_time = float(mat_data['reachability_time'].item())
        conformal_time2 = float(mat_data['conformal_time2'].item())


        Decision = 'true'

        if LB.min() < 0:
            Decision = 'False'

        print(f'The answer is {Decision}')
        Decisions.append(Decision)
        Times.append(training_time+Model_training_time+trn_time1+conformal_time1+conformal_time2+reachability_time+projection_time)

    return Decisions, Times


import scipy.io

mat = scipy.io.loadmat('Bounds.mat')  # Replace with your actual .mat filename

lower = mat['lb']
upper = mat['ub']

K = 20
Nt = 2000
Nc = 100000
ell = 99999
target_class = 88
epochs = 40;

decisions, times = CI(K, Nt, epochs, Nc, ell, target_class, lower, upper)

print("Decisions:", decisions)
print("Times:", times)