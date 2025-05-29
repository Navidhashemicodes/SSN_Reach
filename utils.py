# -*- coding: utf-8 -*-
"""
Created on Thu May 22 11:23:19 2025

@author: navid
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
    
    

# Fixed 12-class color map
colors = np.array([
    [0, 0, 0],         # 0 black
    [128, 0, 0],       # 1 maroon
    [0, 128, 0],       # 2 green
    [128, 128, 0],     # 3 olive
    [0, 0, 128],       # 4 navy
    [128, 0, 128],     # 5 purple
    [0, 128, 128],     # 6 teal
    [255, 255, 255],   # 7 white (ambiguous)
    [255, 0, 0],       # 8 red
    [0, 255, 0],       # 9 lime
    [0, 0, 255],       # 10 blue
    [255, 255, 0],     # 11 yellow
    [255, 165, 0],     # 12 orange
    [75, 0, 130],      # 13 indigo
    [240, 230, 140],   # 14 khaki
    [173, 216, 230],   # 15 lightblue
    [199, 21, 133],    # 16 mediumvioletred
    [0, 100, 0],       # 17 darkgreen
    [0, 191, 255],     # 18 deepskyblue
    [218, 165, 32],    # 19 goldenrod
    [128, 128, 128],   # 20 gray (extra)
    [255, 20, 147],    # 21 deeppink
    [0, 255, 255],     # 22 cyan
    [138, 43, 226],    # 23 blueviolet
    [255, 105, 180],   # 24 hotpink
    [210, 105, 30],    # 25 chocolate
    [244, 164, 96],    # 26 sandybrown
    [32, 178, 170],    # 27 lightseagreen
    [152, 251, 152],   # 28 palegreen
    [135, 206, 250],   # 29 lightskyblue
    [220, 20, 60],     # 30 crimson
    [255, 140, 0],     # 31 darkorange
    [70, 130, 180],    # 32 steelblue
    [255, 228, 196],   # 33 bisque
    [124, 252, 0],     # 34 lawngreen
    [219, 112, 147],   # 35 palevioletred
    [127, 255, 212],   # 36 aquamarine
    [0, 206, 209],     # 37 darkturquoise
    [255, 239, 213],   # 38 papayawhip
    [60, 179, 113],    # 39 mediumseagreen
    [244, 255, 250],   # 40 mintcream
    [176, 224, 230],   # 41 powderblue
    [255, 218, 185],   # 42 peachpuff
    [152, 251, 152],   # 43 palegreen
    [255, 160, 122],   # 44 lightsalmon
    [255, 69, 0],      # 45 orangered
    [72, 209, 204],    # 46 mediumturquoise
    [199, 21, 133],    # 47 mediumvioletred
    [238, 130, 238],   # 48 violet
    [255, 182, 193],   # 49 lightpink
], dtype=np.uint8)

# Use a clearly different color for ambiguous pixels
ambiguous_color = np.array([255, 255, 255], dtype=np.uint8)  # light gray

def plot_bounds_to_mask(classes):
    """
    Visualizes the class map. Ambiguous pixels (with multiple class candidates) are light gray.
    
    Args:
        classes: A 2D list of shape (H, W), where classes[i][j] is a list of class indices.
        
    Returns:
        A (H, W, 3) RGB image as a NumPy array.
    """
    height = len(classes)
    width = len(classes[0])
    masked_img = np.zeros((height, width, 3), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            pixel_classes = classes[i][j]
            if len(pixel_classes) == 1:
                masked_img[i, j] = colors[pixel_classes[0]]
            else:
                masked_img[i, j] = ambiguous_color

    # Display
    plt.imshow(masked_img)
    plt.title("Visualized Class Mask (Ambiguous = Light Gray)")
    plt.axis("off")
    plt.show()

    return masked_img


def plot_logits_to_mask(logits):
    """
    Applies softmax to logits and plots a color-coded segmentation mask.

    Args:
        logits (torch.Tensor): Shape (1, 12, H, W), raw model output.
    """
    
    # Apply softmax and get predicted class for each pixel
    probs = torch.softmax(logits.squeeze(0), dim=0)         # shape: [12, H, W]
    pred_mask = torch.argmax(probs, dim=0).cpu().numpy()    # shape: [H, W]

    # Create color-coded RGB mask
    color_mask = colors[pred_mask]  # shape: (H, W, 3)

    # Plot the mask
    plt.figure(figsize=(10, 7))
    plt.imshow(color_mask)
    plt.title("Predicted Segmentation Mask")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    
    
# def plot_binary_logits_to_mask(pred):
#     """
#     Converts binary logits to a black-and-white segmentation mask.

#     Args:
#         logits (torch.Tensor): Shape (1, 1, H, W), raw model output (pre-sigmoid).
#                                Positive values -> class 1 (white), Negative -> class 0 (black).
#     """
    
    
#     pred = pred.squeeze(1)  # Shape: [1, 304, 304]
#     pred_probs = torch.sigmoid(pred)  # Shape: [1, 304, 304]
#     pred_binary = (pred_probs > 0.45).float()  # Binary mask: [1, 304, 304]
#     pred_binary_np = pred_binary.cpu().detach().numpy()[0]*255  # Shape: [304, 304]
            
#     plt.figure(figsize=(8, 6))
#     plt.imshow(pred_binary_np, cmap='gray', vmin=0, vmax=255)
#     plt.title("Binary Segmentation Mask")
#     plt.axis('off')
#     plt.tight_layout()
#     plt.show()
    
    
    
def plot_binary_logits_to_mask(pred, threshold):
    """
    Converts binary logits to a black-and-white segmentation mask.

    Args:
        logits (torch.Tensor): Shape (1, 1, H, W), raw model output (pre-sigmoid).
                               Positive values -> class 1 (white), Negative -> class 0 (black).
    """
    
    
    pred = pred.squeeze(1)  # Shape: [1, 304, 304]
    pred_binary = (pred > threshold).float()  # Binary mask: [1, 304, 304]
    pred_binary_np = pred_binary.cpu().detach().numpy()[0]*255  # Shape: [304, 304]
            
    plt.figure(figsize=(8, 6))
    plt.imshow(pred_binary_np, cmap='gray', vmin=0, vmax=255)
    plt.title("Binary Segmentation Mask")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
