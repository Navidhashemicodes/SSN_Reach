# from build_BiSeNet import BiSeNet  # assuming this is where your class is
# import torch
#
# # Step 1: Create the model
# model = BiSeNet(num_classes=12, context_path='resnet18')  # use the right num_classes and context_path
#
# # Step 2: Load the weights
# state_dict = torch.load('best_dice_loss_miou_0.655.pth', map_location='cpu')
# model.load_state_dict(state_dict)
#
# # Optional: set model to eval mode if you're not training
# model.eval()
#
# # Step 3: Now you can safely call named_children
# for name, module in model.named_children():
#     print(f"{name}: {module}")
#
#
# torch.save(model, 'BiSeNet_full_model.pth')
#
# dummy_input = torch.randn(1, 3, 720, 960)  # Match model's expected input size
# torch.onnx.export(
#     model,
#     dummy_input,
#     "BiSeNet.onnx",
#     export_params=True,
#     opset_version=11,  # You can try 12 or 13 depending on MATLAB support
#     do_constant_folding=True,
#     input_names=['input'],
#     output_names=['output'],
#     dynamic_axes={
#         'input': {0: 'batch_size'},
#         'output': {0: 'batch_size'}
#     }
# )


import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from build_BiSeNet import BiSeNet
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# Paths
dir1 = './camvid_images'  # <-- Set this to your directory containing .png images
img_name = '0016E5_07959.png'  # <-- Example CamVid image filename

# Step 1: Create the model
model = BiSeNet(num_classes=12, context_path='resnet18')

# Step 2: Load the weights
state_dict = torch.load('best_dice_loss_miou_0.655.pth', map_location='cpu')
model.load_state_dict(state_dict)
model.eval()

# Step 3: Save full model and ONNX
torch.save(model, 'BiSeNet_full_model.pth')

dummy_input = torch.randn(1, 3, 720, 960)
torch.onnx.export(
    model,
    dummy_input,
    "BiSeNet.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

# -----------------------
# Step 4: Inference on an image
# -----------------------
img_path = 'C:/Users/navid/Documents/MATLAB/MATLAB_prev/others/Files/CDC2023/Large_DNN/Case_study/CamVid/Pytorch_repo/CamVid/train/Seq05VD_f03900.png'
# Load and preprocess image
# img_path = os.path.join(dir1, img_name)
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (960, 720))  # Width x Height
img_input = img.astype(np.float32)
img_input = img_input.transpose(2, 0, 1)  # HWC -> CHW
img_input = torch.from_numpy(img_input).unsqueeze(0)  # Add batch dim [1, 3, 720, 960]

# Forward pass
with torch.no_grad():
    output = model(img_input)[0]  # Get the main output (ignore aux outputs)
    output = F.softmax(output, dim=1)  # [1, num_classes, H, W]
    pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()  # [H, W]

# Optional: Define a colormap
colormap = np.array([
    [128, 64, 128],  # Class 0: Road
    [244, 35, 232],  # Class 1: Sidewalk
    [70, 70, 70],    # Class 2: Building
    [102, 102, 156], # Class 3: Wall
    [190, 153, 153], # Class 4: Fence
    [153, 153, 153], # Class 5: Pole
    [250, 170, 30],  # Class 6: Traffic light
    [220, 220, 0],   # Class 7: Traffic sign
    [107, 142, 35],  # Class 8: Vegetation
    [152, 251, 152], # Class 9: Terrain
    [70, 130, 180],  # Class 10: Sky
    [220, 20, 60],   # Class 11: Person
], dtype=np.uint8)

# Map predictions to colors
seg_map = colormap[pred % 12]

# Resize segmentation map to match input image dimensions
seg_map = cv2.resize(seg_map, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

# Plot original image and segmentation
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Input Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(seg_map)
plt.title("Predicted Segmentation")
plt.axis('off')

plt.tight_layout()
plt.show()
