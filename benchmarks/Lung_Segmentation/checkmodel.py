#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 28 14:59:01 2025

@author: hashemn
"""

import torch
import torch.nn as nn
import numpy as np
import onnxruntime as ort
import os
import sys
import pathlib
from PIL import Image
# from torchvision import transforms
import torchvision.transforms as transforms
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
root_dir = pathlib.Path(__file__).resolve().parents[2]

sys.path.append(root_dir)
from utils import plot_binary_logits_to_mask



eval_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


image_name = 'CHNCXR_0005_0.png'
model_name = 'lung_segmentation.onnx'


current_dir = os.getcwd()
model_path = os.path.join(current_dir, 'models', model_name)
image_path = os.path.join(current_dir, 'images', image_name)

img = Image.open(image_path).convert('L')  # 'L' mode = single-channel grayscale
img = eval_transforms(img)
img = img.unsqueeze(0)




from model.unet import UNet

net = UNet(n_channels=1, n_classes=1)

R = torch.load('best_checkpoint.pt')

net.load_state_dict(R['model_state_dict'])
net.eval()
net = net.to(device)



dummy_input = torch.randn(1, 1,512, 512).to(device)

torch.onnx.export(
    net,
    dummy_input,
    model_path,
    verbose=False,
    opset_version=11,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"},
    }
)


ort_session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])

img_tensor = img.reshape(1, 1, 512, 512).to(device, dtype=torch.float32)
img_np = img_tensor.detach().cpu().numpy().astype(np.float32)
at_im = img_np.copy()



# input_names = [input.name for input in ort_session.get_inputs()]

output0 = ort_session.run(None, {'input': img_np})
output0 = torch.tensor(output0[0]).to(device)
output = net(img_tensor)

print(output-output0)

threshold = 0

plot_binary_logits_to_mask(output, threshold)
plot_binary_logits_to_mask(output0, threshold)