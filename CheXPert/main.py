import torch

from pyversion_Case import CheXpert_exp


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
start_loc = (512 - 20, 512 - 20)
Ns = 10
Nsp = 2
rank = 7
guarantee = 0.999
delta_rgb = 3
de = delta_rgb / 255.0
Nt = 10
N_dir = 6
epsilon = 0.001

image_names = [
    'CHNCXR_0005_0.png',
    'MCUCXR_0258_1.png'
    # 'MCUCXR_0264_1.png',
    # 'MCUCXR_0266_1.png',
    # 'MCUCXR_0275_1.png',
    # 'MCUCXR_0282_1.png',
    # 'MCUCXR_0289_1.png',
    # 'MCUCXR_0294_1.png',
    # 'MCUCXR_0301_1.png',
    # 'MCUCXR_0309_1.png',
    # 'MCUCXR_0311_1.png',
    # 'MCUCXR_0313_1.png',
    # 'MCUCXR_0316_1.png',
    # 'MCUCXR_0331_1.png',
    # 'MCUCXR_0334_1.png',
    # 'MCUCXR_0338_1.png',
    # 'MCUCXR_0348_1.png',
    # 'MCUCXR_0350_1.png',
    # 'MCUCXR_0352_1.png',
    # 'MCUCXR_0354_1.png'
]

# N_perturbed_list = [17, 34, 51, 68, 85, 102]
N_perturbed_list = [17, 34]

ii=0
for idx, image_name in enumerate(image_names):
    for N_perturbed in N_perturbed_list:
        
        print(f"Running: {image_name} with N_perturbed={N_perturbed}")
    
        ii = ii+1
        CheXpert_exp(
            start_loc,
            N_perturbed,
            delta_rgb,
            image_name,
            Nt,
            N_dir,
            Ns,
            Nsp,
            rank,
            guarantee,
            device
        )