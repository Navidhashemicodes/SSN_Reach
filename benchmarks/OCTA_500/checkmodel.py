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
from torchvision import transforms
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
root_dir = pathlib.Path(__file__).resolve().parents[2]

sys.path.append(root_dir)
from utils import plot_binary_logits_to_mask

from unet import UNet
import matplotlib.pyplot as plt



def convert_batchnorm_to_instancenorm(model):
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            num_features = module.num_features
            inst_norm = nn.InstanceNorm2d(num_features, affine=True)
            setattr(model, name, inst_norm)
        else:
            convert_batchnorm_to_instancenorm(module)
    return model




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


image_name = '10491.bmp'
model_name = 'betti_best.onnx'


current_dir = os.getcwd()
model_path = os.path.join(current_dir, 'models', model_name)
image_path = os.path.join(current_dir, 'images', image_name)


img = Image.open(image_path)
to_tensor = transforms.ToTensor()
img = to_tensor(img)

from unet import UNet
net = UNet(in_channels=1, n_classes=1, channels=128)
net.load_state_dict(torch.load("betti_best.pth", map_location=device))

net = convert_batchnorm_to_instancenorm(net)

net = net.to(device)
net.eval()


dummy_input = torch.randn(1, 1, 304, 304).to(device)

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


# dummy_input = torch.randn(1, 1, 304, 304).to(device)  # Same input shape used in the inference
# torch.onnx.export(net, dummy_input, "betti_best.onnx", verbose=True, opset_version=11)


ort_session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])

img_tensor = img.reshape(1, 1, 304, 304).to(device, dtype=torch.float32)
img_np = img_tensor.detach().cpu().numpy().astype(np.float32)
at_im = img_np.copy()



# input_names = [input.name for input in ort_session.get_inputs()]

output0 = ort_session.run(None, {'input': img_np})
output0 = torch.tensor(output0[0]).to(device)
output = net(img_tensor)

print(output-output0)

threshold = np.log(45/55)
plot_binary_logits_to_mask(output, threshold)
plot_binary_logits_to_mask(output0, threshold)