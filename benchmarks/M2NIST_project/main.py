# -*- coding: utf-8 -*-
"""
Created on Wed May 21 23:57:52 2025

@author: navid
"""

import torch
import numpy as np
import onnxruntime as ort
import cv2
import os
import sys
import pathlib
import scipy.io

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
root_dir = pathlib.Path(__file__).resolve().parents[2]
reach_factory_path = os.path.join(root_dir, 'Reach_Factory')
sys.path.append(reach_factory_path)
from Reach4SSN import ReachabilityAnalyzer


def M2NIST_exp( start_loc, N_perturbed, de, image_number, Nt, N_dir,
                  Ns, Nsp, rank, guarantee, device,  threshold_normal,
                  sim_batch, trn_batch, epochs, surrogate_mode, src_dir, nnv_dir):
        
    model_name = 'm2nist_dilated_72iou_24layer.onnx'
    image_name = 'm2nist_6484_test_images.mat'

    current_dir = os.getcwd()
    model_path = os.path.join(current_dir, 'models', model_name)
    image_path = os.path.join(current_dir, 'images', image_name)
    
    # Load MATLAB data
    mat = scipy.io.loadmat(image_path)
    images = mat['im_data']  # (64, 84, 1000)
    image = images[:,:,image_number]
    image = image[:, :, np.newaxis]

    # Convert shape and type
    image_np = np.transpose(image, (2, 0, 1))      # (1, 64, 84)
    image_np = image_np[:, np.newaxis, :, :].astype(np.float32)  # (1, 1, 64, 84)


    ort_session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
    
    
    at_im = image_np.copy()  # Also shape: [1, 3, 720, 960]

    ct = 0
    indices = []
    _, _, H, W = image_np.shape     # Shape: [1, 3, 720, 960]

    for i in range(start_loc[0], H):
        for j in range(start_loc[1], W):
            if np.min(image_np[0, :, i, j]) > 150:
                at_im[0, :, i, j] = 0.0  # Reset all 3 channels at once
                indices.append([i, j])
                ct += 1
                if ct == N_perturbed:
                    print(f"{N_perturbed} pixels found.")
                    break
        if ct == N_perturbed:
            break

    indices = np.array(indices)


    image_tensor = torch.from_numpy(image_np).to(device)
    x = image_tensor.to(torch.float16)  # Use half precision
    x_numpy = x.cpu().numpy().astype(np.float32)
    output = ort_session.run(None, {'input': x_numpy})
    output = torch.tensor(output[0]).to(device)
    
    output_dim = output.shape

    _, True_class_tensor = torch.max(output, dim=1)  
    True_class_tensor = True_class_tensor.squeeze(0) 
    True_class = True_class_tensor.cpu().tolist()


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
        original_dim = (1, 64, 84),
        output_dim = output_dim,
        device=device,
        mode = surrogate_mode,
        class_threshold = 0,
        src_dir = src_dir,
        nnv_dir = nnv_dir,
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




if __name__ == '__main__':
    
    
    # The following hyperparameters can be adjusted to fit the verification process
    # within the limits of your GPU memory. The current values are according to experiments
    # in the submission

    # Ns: Number of calibration samples. Increasing Ns can improve the level of formal guarantees.

    # Nsp: Number of calibration samples processed per iteration on the GPU. Adjust this value
    # to ensure that the per-iteration data fits in memory. Higher values reduce runtime,
    # but increase GPU memory requirements.

    # Nt: Number of training samples to be loaded onto the GPU in full. A larger Nt typically
    # leads to tighter (less conservative) bounds, but requires more memory. On GPUs with
    # limited memory, you may need to reduce Nt, accepting slightly more conservative results.
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    start_loc = (0, 0)
    Ns = 100000
    Nsp = 10000
    rank = 99999
    guarantee = 0.9999
    delta_rgb = 3
    de = delta_rgb
    Nt = 10000
    N_dir = 5000
    threshold_normal = 1e-5
    sim_batch = 1000
    trn_batch = 1000
    epochs = 90
    surrogate_mode = 'ReLU'
    src_dir = os.path.join(root_dir, 'src')
    nnv_dir = 'C:\\Users\\navid\\Documents\\nnv'
    
    if not os.path.isdir(nnv_dir):
        sys.exit(f"❌ Error: NNV directory not found at '{nnv_dir}'.\n"
                 f"Please check the path and ensure NNV is properly installed.")

    print(f"✅ NNV directory found at: {nnv_dir}")


    # image_number_list = [ 0 ,1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    #                 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    # N_perturbed_list = [17, 34, 51, 68, 85, 102]
    
    image_number_list = [ 0 ,1 ]
    N_perturbed_list = [17, 34]
    
    ii=0
    for idx, image_number in enumerate(image_number_list):
        for N_perturbed in N_perturbed_list:
            
            print(f"Running: image number: {image_number} with N_perturbed={N_perturbed}")
            
            ii = ii+1
            M2NIST_exp( start_loc, N_perturbed, de, image_number, Nt, N_dir,
                          Ns, Nsp, rank, guarantee, device,  threshold_normal,
                          sim_batch, trn_batch, epochs, surrogate_mode, src_dir, nnv_dir)