import torch
import numpy as np
import onnxruntime as ort
import time
from torch.cuda.amp import autocast

def CI(K, Nt, Nc, ell, target_class, lower, upper):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epsilon = 0.001

    # Load ONNX model with GPU provider
    model_path = 'CIFAR100_resnet_large.onnx'
    ort_session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])

    #def CIFAR100Func(x):
     #   x_numpy = x.cpu().numpy().astype(np.float32)
      #outputs = ort_session.run(None, {'modelInput': x_numpy})
       # return torch.tensor(outputs[0]).to(device).T  # [num_classes x batch_size]

    

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

        del X
        torch.cuda.empty_cache()

        start_time = time.time()
        Y_centered = Y - Y.mean(dim=1, keepdim=True)
        SigmaY = (Y_centered @ Y_centered.T) / (Y.shape[1] - 1)
        eigvals, VY = torch.linalg.eigh(SigmaY)

        C = 20 * (epsilon * Y.mean(dim=1) + (0.05 - epsilon) * 0.5 * (Y.min(dim=1).values + Y.max(dim=1).values))
        dY = Y - C.unsqueeze(1)
        dYV = VY.T @ dY
        r_max = torch.max(torch.abs(dYV), dim=1).values
        training_time = time.time() - start_time

        del Y, dYV, dY
        torch.cuda.empty_cache()

        lbk = lbb.unsqueeze(0).repeat(Nc, 1, 1, 1)
        ubk = ubb.unsqueeze(0).repeat(Nc, 1, 1, 1)
        dbk = ubk - lbk

        torch.manual_seed(1)
        start_time = time.time()
        X_test = lbk + torch.rand_like(dbk) * dbk
        Y_test = CIFAR100Func(X_test)
        test_data_generation_time = time.time() - start_time

        del X_test
        torch.cuda.empty_cache()

        start_time = time.time()
        dY_test = Y_test - C.unsqueeze(1)
        dYV_test = VY.T @ dY_test
        Rs = torch.max(torch.abs(dYV_test) / r_max.unsqueeze(1), dim=0).values
        Rs_sorted, _ = torch.sort(Rs)
        R_star = Rs_sorted[ell - 1]  # ell is 1-based

        d_lb = -R_star * r_max
        d_ub = R_star * r_max
        conformal_time = time.time() - start_time

        del Y_test, dYV_test, dY_test
        torch.cuda.empty_cache()

        start_time = time.time()
        Decision = 'true'

        C1 = C[target_class] - C
        VY1 = VY[target_class, :] - VY

        LB = C1 + 0.5 * (VY1 + VY1.abs()) @ d_lb + 0.5 * (VY1 - VY1.abs()) @ d_ub

        if LB.min() < 0:
            Decision = 'False'

        verification_time = time.time() - start_time

        print(f'The answer is {Decision}')
        Decisions.append(Decision)
        Times.append(conformal_time + test_data_generation_time +
                     training_time + train_data_generation_time + verification_time)

    return Decisions, Times


import scipy.io

mat = scipy.io.loadmat('Bounds.mat')  # Replace with your actual .mat filename

lower = mat['lb']
upper = mat['ub']

K = 20
Nt = 1
Nc = 2000
ell = 1999
target_class = 79-1

decisions, times = CI(K, Nt, Nc, ell, target_class, lower, upper)

print("Decisions:", decisions)
print("Times:", times)