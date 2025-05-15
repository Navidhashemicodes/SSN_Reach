import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import onnxruntime as ort
import time
from torch.cuda.amp import autocast
import os
import matlab.engine
import scipy.io

# for PGD
from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier
from onnx2torch import convert


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
        
        from Training_Linear import Trainer_Linear
        current_dir = os.getcwd()
        save_path = os.path.join(current_dir, 'trained_A_b.mat')

        Map, Model_training_time = Trainer_Linear(X, dYV, device, epochs, save_path)

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
        X_test_copy = X_test.clone().detach()
        X_test_copy = X_test_copy.cpu().numpy()
        Y_test = CIFAR100Func(X_test)
        test_data_generation_time = time.time() - start_time

        print('Starting model conversion...')

        torch_model = convert(model_path).eval().to('cpu')
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(torch_model.parameters(), lr=0.001)
        # add PGD samples
        classifier = PyTorchClassifier(
            model=torch_model,
            loss=loss_fn,
            optimizer=optimizer,
            clip_values=(0.0, 1.0),
            input_shape=(3, 32, 32),
            nb_classes=100
        )
        attack = ProjectedGradientDescent(
            estimator=classifier,
            norm=np.inf,
            eps=0.3,
            eps_step=0.01,
            max_iter=40,
            targeted=False
        )
        print('Starting attack generation...')
        x_test_adv = attack.generate(x=X_test_copy[:500])
        print(x_test_adv.shape)
        print('PGD samples done')
        x_test_adv = torch.from_numpy(x_test_adv)
        x_test_adv = x_test_adv.to(X_test.device).type_as(X_test)
        print(f'Original X_test shape: {X_test.shape}')

        Y_test_adv = CIFAR100Func(x_test_adv, batch_size=X_test.shape[0])

        X_test = torch.cat((X_test, x_test_adv), dim=0)
        print(f'New X_test shape: {X_test.shape}')

        X_test = X_test.view(X_test.shape[0] , -1).T
        print(f'New Y_test shape: {Y_test.shape}')
        print(f'New Y_test_adv shape: {Y_test_adv.shape}')

        Y_test = torch.cat((Y_test, Y_test_adv), dim=1)
        print(f'New Y_test shape: {Y_test.shape}')
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
        print(eng.sqrt(16.0))  # Should print 4.0


        matlab_code = r"""


        clear
        clc

        load('python_data.mat')
        load('trained_A_b.mat')
        load('Bounds.mat')
        A = double(W);
        b = double(b)';

        addpath(genpath('C:\Users\navid\Documents\MATLAB\MATLAB_prev\others\Files\CDC2023\Large_DNN\src'))
        addpath(genpath('C:\Users\navid\Documents\nnv'))

        dim2 = 100;
        dimp = 3072;

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