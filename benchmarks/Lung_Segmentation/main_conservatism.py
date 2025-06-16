# -*- coding: utf-8 -*-
"""
Created on Wed May 21 23:57:52 2025

@author: navid
"""

import torch
import numpy as np
import onnxruntime as ort
import cv2
from PIL import Image
import os
import sys
import pathlib
import torchvision.transforms as transforms


sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
root_dir = pathlib.Path(__file__).resolve().parents[2]
reach_factory_path = os.path.join(root_dir, 'Reach_Factory')
sys.path.append(reach_factory_path)
from HowConservative import Conservatism_analysis


def LS_conservatism( start_loc, N_perturbed, delta_rgb, image_name, Ns, Nsp, device, sim_batch, SEED0):
    
    
    
    de = delta_rgb / 255.0
    model_name = 'lung_segmentation.onnx'
    
    
    base_name = os.path.splitext(image_name)[0]

    Result_name = f"CI_result_middle_guarantee_ReLU_relaxed_eps_{delta_rgb}_Npertubed_102_{base_name}.pt"

    current_dir = os.getcwd()
    model_path = os.path.join(current_dir, 'models', model_name)
    image_path = os.path.join(current_dir, 'images', image_name)
    Result_path = os.path.join(current_dir, 'Results_e', Result_name)

    ort_session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])

    eval_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])


    img = Image.open(image_path).convert('L')  # 'L' mode = single-channel grayscale
    img = eval_transforms(img)
    img_tensor = img.unsqueeze(0).to(device)
    img_np = img_tensor.detach().cpu().numpy().astype(np.float32)

    at_im = img_np.copy()
    
    x = img_tensor.to(torch.float16)  # Use half precision
    x_numpy = x.cpu().numpy().astype(np.float32)
    output = ort_session.run(None, {'input': x_numpy})
    output = torch.tensor(output[0]).to(device)

    output_dim = output.shape
    
    Data = torch.load(Result_path)
    
    
    
    
    
    
    # --- Perturbation loop ---
    ct = 0
    indices = []
    _, _, H, W = img_np.shape     # Shape: [1, 3, 720, 960]

    for i in range(start_loc[0], H):
        for j in range(start_loc[1], W):
            if np.min(img_np[0, :, i, j]) > 150 / 255.0:
                at_im[0, :, i, j] = 0.0  # Reset all 3 channels at once
                indices.append([i, j])
                ct += 1
                if ct == N_perturbed:
                    print(f"{N_perturbed} pixels found.")
                    break
        if ct == N_perturbed:
            break

    indices = np.array(indices)
    # indices = Data['indices']
    
    at_im_tensor = torch.from_numpy(at_im).to(device)
    
    

    LBB = Data['Lb_pixels']
    UBB = Data['Ub_pixels']
    
    height, width, n_class = output_dim[2], output_dim[3], output_dim[1]
    Lb_temp = LBB.view(1, height, width, n_class)
    Ub_temp = UBB.view(1, height, width, n_class)

    LB_out = Lb_temp.permute(0, 3, 1, 2).contiguous()
    UB_out = Ub_temp.permute(0, 3, 1, 2).contiguous()
    LB_out = LB_out.view(LB_out.shape[0], -1).to(device)
    UB_out = UB_out.view(UB_out.shape[0], -1).to(device)
   


    params = {
        'N_perturbed' : N_perturbed,
        'image_name' : image_name,
        'delta_rgb': delta_rgb,
        'Ns' : Ns,
        'Nsp' : Nsp,
        'sim_batch' : sim_batch,
        'device' : device,
        'SEED0' : SEED0,
        'input_name' : 'input'
    }
    conservatism_analyzer = Conservatism_analysis(
        model = ort_session,
        LB = at_im_tensor,
        de = de,
        LB_out = LB_out,
        UB_out = UB_out,
        indices = indices,
        original_dim = (1, 512, 512),
        device=device,
        params=params
    )
    emprical_miscoverage, Y_min, Y_max = conservatism_analyzer.conservatism()

    return emprical_miscoverage, Y_min, Y_max




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
    Ns = 1000000
    Nsp = 1000
    sim_batch = 50
    SEED0 = 1000
    

    image_name = 'CHNCXR_0005_0.png'
    
    Results_eps = []
    Results_LB_out = []
    Results_UB_out = []
    
    N_perturbed =  102
    delta_rgb = 10    
                
    print(f"Running: {image_name} with N_perturbed={N_perturbed}")
            
    emprical_miscoverage, Y_min, Y_max = LS_conservatism( start_loc, N_perturbed, delta_rgb, image_name, Ns, Nsp, device, sim_batch, SEED0)
            
            
    Results_eps.append(emprical_miscoverage)
    Results_LB_out.append(Y_min)
    Results_UB_out.append(Y_max)
    
    
    
    
    N_perturbed =  102
    delta_rgb = 150    
                
    print(f"Running: {image_name} with N_perturbed={N_perturbed}")
            
    emprical_miscoverage, Y_min, Y_max = LS_conservatism( start_loc, N_perturbed, delta_rgb, image_name, Ns, Nsp, device, sim_batch, SEED0)
            
            
    Results_eps.append(emprical_miscoverage)
    Results_LB_out.append(Y_min)
    Results_UB_out.append(Y_max)
    
    
    
    torch.save(  { 'Results_eps' : Results_eps , 'Results_LB_out' : Results_LB_out , 'Results_UB_out' : Results_UB_out }  , 'Results.pt'   )