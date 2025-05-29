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
# from tqdm import tqdm
from scipy.stats import beta
# from torch.cuda.amp import autocast
import matlab.engine
import scipy.io
import pathlib

root_dir = pathlib.Path(__file__).resolve().parents[1]
training_factory_path = os.path.join(root_dir, 'Training_Factory')
sys.path.append(training_factory_path)

from Direction_training import compute_directions
from Training_ReLU import Trainer_ReLU
from Training_Linear import Trainer_Linear

class ReachabilityAnalyzer:
    
    
    def __init__(self, True_class, class_threshold, model, image_name, LB, de, indices, original_dim, output_dim, device, mode, src_dir, nnv_dir, params):
        
        self.True_class = True_class
        self.class_threshold = class_threshold
        self.de = de
        self.indices = indices
        self.device = device
        self.model = model
        self.LB = LB
        self.original_dim = original_dim
        self.output_dim = output_dim
        self.mode = mode
        self.image_name = image_name
        self.src_dir = src_dir
        self.nnv_dir = nnv_dir
        self.params = params
        
        
    def mat_generator_no_third(self, repeat, values):
        
        N_perturbed = self.params['N_perturbed']
        
        Matrix = torch.zeros( (repeat, *self.original_dim), device=values.device, dtype=values.dtype)
        
        t = 0
        for c in range(self.original_dim[0]):
            for i in range(N_perturbed):
                row, col = self.indices[i]
                Matrix[:,c,row, col] = values[:,t]
                t += 1
        return Matrix
    
    
    def Func(self, x):
        batch_size = self.params['sim_batch']
        # x = x.to(torch.float16)  # Use half precision
        x_numpy = x.cpu().numpy().astype(np.float32)
        results = []
        for i in range(0, x_numpy.shape[0], batch_size):
            batch = x_numpy[i:i+batch_size]
            #with autocast():  # Automatically use mixed precision
            with torch.amp.autocast('cuda'):
                output = self.model.run(None, {'input': batch})
            results.append(torch.tensor(output[0]).to(self.device))
        return torch.cat(results, dim=0)
    
    
    def generate_data_chunk(self, repeat, LBs):
        
        N_perturbed = self.params['N_perturbed']
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
    
    
    def CI_ReLU_surrogate(self, small_net, C, res_max, Directions):
        
        Ns = self.params['Ns']
        Nsp = self.params['Nsp']        

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
            
            Y_test, X_test_nc, tst_run = self.generate_data(curr_len, nc+1)
            test_data_run.append( tst_run )
            
            t1 = time()
            with torch.no_grad():
                pred = small_net(X_test_nc) @ Directions  + C.unsqueeze(0)  # shape: (dim2, curr_len)
            res_tst = (Y_test - pred).abs()
            Rs[ind:ind + curr_len] = torch.max(res_tst / res_max.unsqueeze(0), dim=1).values
            res_test_time.append(time() - t1)
            
            del Y_test,  X_test_nc, res_tst
            ind += curr_len
        
        t0 = time()

        with torch.no_grad():
            Rs_sorted = torch.sort(Rs).values
            R_star = Rs_sorted[ell]  # Assuming `ell` is defined
            Conf = R_star * res_max

        conformal_time = time() - t0
            
        return Conf, R_star, conformal_time, res_test_time, test_data_run
    
    
    def call_MATLAB_for_star(self):
    
        eng = matlab.engine.start_matlab()


        matlab_code = r"""


        clear
        clc
        
        load('python_data.mat')
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

        dim2 = length(Conf);

        tic
        H.V = [sparse(dim2,1) speye(dim2)];
        H.C = sparse(1,dim2);
        H.d = 0;
        H.predicate_lb = -Conf.';
        H.predicate_ub =  Conf.';
        H.dim = dim2;
        H.nVar= dim2;
        %%%%%%%%%%%%%%%
        conformal_time2 = toc;
       
        
       
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
        Surrogate_reach = affineMap(Principal_reach , Directions.' , C.');

        %%%%%%%%%%%%%%
        reachability_time = toc;
        clear Principal_reach

        tic
        %%%%%%%%%%%%%%
        P_Center = sparse(double(Surrogate_reach.V(:,1)));
        P_lb = double([Surrogate_reach.predicate_lb ; H.predicate_lb]);
        P_ub = double([Surrogate_reach.predicate_ub ; H.predicate_ub]);


        P_V = [double(Surrogate_reach.V(:,2:end))   double(H.V(:,2:end))];
        Lb = P_Center + 0.5*(P_V + abs(P_V))*P_lb + 0.5*(P_V - abs(P_V))*P_ub;
        Ub = P_Center + 0.5*(P_V + abs(P_V))*P_ub + 0.5*(P_V - abs(P_V))*P_lb;


        Lb_pixels = reshape(Lb , [dimp, dim2/dimp]);
        Ub_pixels = reshape(Ub , [dimp, dim2/dimp]);
        %%%%%%%%%%%%%%
        projection_time = toc;

        clear  Surrogate_reach

        save("Matlab_data.mat", 'Lb_pixels', 'Ub_pixels', 'projection_time', 'reachability_time', 'conformal_time2')

        """
        
        eng.eval(matlab_code, nargout=0)
        eng.quit()

    
    
    def Verify_with_ReLU_surrogate(self):
        
        Nt = self.params['Nt']
        N_dir = self.params['N_dir']
       
        
        assert N_dir <= Nt, "Requested more samples than available!"

        selected_indices = torch.randperm(Nt)[:N_dir]
        
        Y, X, train_data_run_1 = self.generate_data(Nt, 0)
        
        X_dir = X[selected_indices, ...].to(self.device)
        Y_dir = Y[selected_indices, ...].to(self.device)
        
        

        Directions, Direction_Training_time = compute_directions(Y_dir, self.device, self.params['trn_batch'])

        Directions = torch.stack([d.squeeze(-1) for d in Directions])


        C = 20 * (0.001 * Y.mean(dim=0) + (0.05 - 0.001) * 0.5 * (Y.min(dim=0).values + Y.max(dim=0).values))
        dY = Y - C.unsqueeze(0)
        dYV = dY @ Directions.T

        torch.cuda.empty_cache()


        
        current_dir = os.getcwd()
        save_path = os.path.join(current_dir, 'trained_relu_weights_2h_norm.mat')
        small_net, Model_training_time = Trainer_ReLU(X, dYV, self.device, self.params['epochs'], save_path)
            
        with torch.no_grad():
            pred = small_net(X) 

            approx_Y = pred @ Directions  + C.unsqueeze(0)  # shape: same as Y

        t0 = time()
        residuals = (Y - approx_Y).abs()
        res_max = residuals.max(dim=0).values
        tn = self.params['threshold_normal']
        res_max[res_max < tn ] = tn
        
        del Y, approx_Y, pred
        
        trn_time1 = time()-t0

        
        Conf, R_star, conformal_time, res_test_time, test_data_run = self.CI_ReLU_surrogate(small_net, C, res_max, Directions)


        current_dir = os.getcwd()
        save_path = os.path.join(current_dir, 'python_data.mat')
        c = C.cpu().numpy()
        conf = Conf.cpu().numpy()
        directions = Directions.cpu().numpy()
        
        scipy.io.savemat(save_path, {
            'Conf': conf, 'C': c, 'Directions': directions, 'dir': self.src_dir,
            'dimp': self.original_dim[1]*self.original_dim[2], 'nnv_dir': self.nnv_dir} )

        del conf, c, directions
    
        self.call_MATLAB_for_star()
    
        current_dir = os.getcwd()
        mat_file_path = os.path.join(current_dir, 'Matlab_data.mat')
        mat_data = scipy.io.loadmat(mat_file_path)

        Lb_pixels = torch.tensor(mat_data['Lb_pixels'], dtype=torch.float32)
        Ub_pixels = torch.tensor(mat_data['Ub_pixels'], dtype=torch.float32)
        projection_time = float(mat_data['projection_time'].item())
        reachability_time = float(mat_data['reachability_time'].item())
        conformal_time2 = float(mat_data['conformal_time2'].item())
        
        return (Lb_pixels, Ub_pixels, Conf, R_star, res_max, projection_time, reachability_time, conformal_time2,
               conformal_time, res_test_time, test_data_run , trn_time1, Direction_Training_time, 
               Model_training_time, train_data_run_1)

    
    def Mask_titles(self):
        
        N_perturbed = self.params['N_perturbed']
        Ns = self.params['Ns']
        guarantee = self.params['guarantee']
        ell = self.params['rank']
        Failure_chance_of_guarantee = beta.cdf(guarantee, ell, Ns + 1 - ell)
        
        
        if self.mode == 'ReLU':
            (Lb_pixels, Ub_pixels, Conf, R_star, res_max, projection_time, reachability_time, conformal_time2, 
            conformal_time, res_test_time, test_data_run , trn_time1, Direction_Training_time,
            Model_training_time, train_data_run_1) = self.Verify_with_ReLU_surrogate()
        elif self.mode == 'Linear':
            (Lb_pixels, Ub_pixels, Conf, R_star, res_max, projection_time, reachability_time, conformal_time2, 
            conformal_time, res_test_time, test_data_run , trn_time1, Direction_Training_time,
            Model_training_time, train_data_run_1) = self.Verify_with_Linear_surrogate()
        elif self.mode == 'Naive':
            (Lb_pixels, Ub_pixels, Conf, R_star, res_max, projection_time, reachability_time, conformal_time2, 
            conformal_time, res_test_time, test_data_run , trn_time1, Direction_Training_time,
            Model_training_time, train_data_run_1) = self.Verify_with_no_surrogate()
        
        
        
        
        start_time = time()
        
        height = self.output_dim[2]
        width= self.output_dim[3]
        mask_dim = self.output_dim[1]
        
        if mask_dim == 1:
            
            # Some SSNs with two classes have one dimensional logits
            # where each class is found using a threshold on this logit
            
            classes = [[None for _ in range(height)] for _ in range(width)]
    
            for i in range(height):
                for j in range(width):
                    t = i * width + j  
                    lb = Lb_pixels[t].item()
                    ub = Ub_pixels[t].item()
                    
                    if lb > self.class_threshold:
                        class_members = [1]
                    elif ub <= self.class_threshold:
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

        for i in range(height):
            for j in range(width):
                if len(classes[i][j]) == 1:
                    if classes[i][j] == [self.True_class[i][j]]:
                        robust += 1
                    else:
                        nonrobust += 1
                else:
                    if self.True_class[i][j] in classes[i][j]:
                        unknown += 1
                    else:
                        nonrobust += 1

        # Compute the robustness percentage
        dim_pic = height*width
        RV = 100 * robust / dim_pic
        
        print(f"Number of Robust pixels: {robust}")
        print(f"Number of non-Robust pixels: {nonrobust}")
        print(f"Number of unknown pixels: {unknown}")
        print(f"RV value: {RV}")
        

        
        print(f"Pr[RV value > {guarantee}%] > {1 - Failure_chance_of_guarantee}")


        verification_runtime = train_data_run_1 + trn_time1 + sum(test_data_run) + sum(res_test_time) + \
                               conformal_time + reachability_time + projection_time + Direction_Training_time + Model_training_time

        print(f"The verification runtime is: {verification_runtime / 60:.2f} minutes.")


        save_dict = {
            "robust": robust,
            "nonrobust": nonrobust,
            "attacked": attacked,
            "unknown": unknown,
            "True_class": self.True_class,
            "classes": classes,
            "Conf": Conf,
            "Nt": self.params['Nt'],
            "N_dir": self.params['N_dir'],
            "de": self.de,
            "ell": ell,
            "Lb_pixels": Lb_pixels,
            "Ub_pixels": Ub_pixels,
            "Ns": Ns,
            "R_star": R_star,
            "res_max": res_max,
            "RV": RV,
            "verification_runtime": verification_runtime,
            "threshold_normal": self.params['threshold_normal'],
            "train_data_run_1": train_data_run_1,
            "trn_time1": trn_time1,
            "test_data_run": test_data_run,
            "res_test_time": res_test_time,
            "conformal_time": conformal_time,
            "reachability_time": reachability_time,
            "projection_time": projection_time,
            "Direction_Training_time": Direction_Training_time,
            "Model_training_time": Model_training_time
            }


        for key, val in save_dict.items():
            if isinstance(val, torch.Tensor):
                save_dict[key] = val.cpu()
            elif isinstance(val, list):
                save_dict[key] = [v.cpu() if isinstance(v, torch.Tensor) else v for v in val]
            elif isinstance(val, dict):
                save_dict[key] = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in val.items()}
        
        # save_name = f"CI_result_middle_guarantee_ReLU_relaxed_eps_{delta_rgb}_Npertubed_{N_perturbed}"+image_name+".pt"
        base_name = os.path.splitext(self.image_name)[0]
        save_name = f"CI_result_middle_guarantee_ReLU_relaxed_eps_{self.de}_Npertubed_{N_perturbed}_{base_name}.pt"
        torch.save(save_dict, save_name)
        
        
        print('All the details are saved')


        
if __name__ == '__main__':
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    start_loc = (0, 0)
    Ns = 10
    Nsp = 2
    rank = 7
    guarantee = 0.999
    delta_rgb = 150
    de = delta_rgb / 255.0
    Nt = 10
    N_dir = 6
    threshold_normal = 1e-5
    sim_batch = 2
    trn_batch = 5
    epochs = 50
    N_perturbed = 17
    surrogate_mode = 'ReLU'
    
        
    
   
    image_name = 'CHNCXR_0005_0.png'
    model_name = 'lung_segmentation.onnx'
    print(f"Running: {image_name} with N_perturbed={N_perturbed}")
    
   
    current_dir = os.getcwd()

    repo_root = os.path.abspath(os.path.join(current_dir, '..'))

    model_path = os.path.join(repo_root, 'benchmarks', 'Lung_Segmentation', 'models', model_name)
    image_path = os.path.join(repo_root, 'benchmarks', 'Lung_Segmentation', 'images', image_name)
    src_dir =  os.path.join(repo_root, 'src')

    ort_session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # ensures it's grayscale
    img = cv2.resize(img, (512, 512))
    img = img.astype(np.float32) / 255.0
    img = img.reshape(1, 1, 512, 512)
    at_im = img.copy()
    
    
    img_tensor = torch.from_numpy(img).to(device)
    # x = img_tensor.to(torch.float16)  # Use half precision
    x_numpy = img_tensor.cpu().numpy().astype(np.float32)
    output = ort_session.run(None, {'input': x_numpy})
    output = torch.tensor(output[0]).to(device)
    
    output_dim = output.shape
    
    output_np = output.squeeze().cpu().numpy()  # shape: [512, 512]

    True_class = [[int(output_np[i, j] > 0) for j in range(512)] for i in range(512)]

    # --- Apply darkening attack ---
    ct = 0
    indices = []

    _, _, H, W = img.shape
    for i in range(start_loc[0], H):
        for j in range(start_loc[1], W):
            if np.min(img[:,:,i, j]) > 150 / 255.0:
                at_im[:,:,i, j] = 0.0
                indices.append([i, j])
                ct += 1
                if ct == N_perturbed:
                    print(f"{N_perturbed} pixels found.")
                    break
        if ct == N_perturbed:
            break

    indices = np.array(indices)
    at_im_tensor = torch.from_numpy(at_im).to(device)
    params = {
        'N_perturbed' : N_perturbed,
        'de' : de,
        'image_name' : image_name,
        'Nt' : Nt,
        'N_dir' : N_dir,
        'Ns' : Ns,
        'Nsp' : Nsp,
        'rank' : rank,
        'guarantee': guarantee,
        'threshold_normal' : threshold_normal,
        'trn_batch' : trn_batch,
        'sim_batch' : sim_batch,
        'epochs' : epochs,
        'device' : device,
    }
    analyzer = ReachabilityAnalyzer(
        True_class = True_class,
        model = ort_session,
        image_name = image_name,
        LB = at_im_tensor,
        de = de,
        indices = indices,
        original_dim = (1, 512, 512),
        output_dim = output_dim,
        device=device,
        mode = surrogate_mode,
        class_threshold = 0,
        src_dir = src_dir,
        params=params
    )
    analyzer.Mask_titles()
    
    remove_path =  os.path.join(current_dir, 'Matlab_data.mat')
    os.remove(remove_path)
    remove_path =  os.path.join(current_dir, 'python_data.mat')
    os.remove(remove_path)
    if surrogate_mode == 'ReLU':
        remove_path =  os.path.join(current_dir, 'trained_relu_weights_2h_norm.mat')
        os.remove(remove_path)