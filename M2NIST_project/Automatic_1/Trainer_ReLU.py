import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import scipy.io
import numpy as np
import random
import time

def estimate_lipschitz(x, y, num_samples=1000):
    n = x.shape[0]
    print(n)
    slopes = []

    for _ in range(num_samples):
        i, j = random.sample(range(n), 2)
        diff_x = x[i] - x[j]
        diff_y = y[i] - y[j]

        norm_x = torch.norm(diff_x)
        norm_y = torch.norm(diff_y)

        if norm_x > 1e-8:  # avoid division by near-zero
            slope = norm_y / norm_x
            slopes.append(slope.item())

    return max(slopes)


# Define the path to the .mat file
mat_file_path = r"/home/hashemn/Documents/Navid/M2NIST_project/Automatic_1/Reduced_dimension.mat"

# Load MATLAB data
mat_data = scipy.io.loadmat(mat_file_path)

# Extract and transpose data for PyTorch
x = torch.tensor(mat_data['X'].T, dtype=torch.float32)  # Shape [10000, 5376]
y = torch.tensor(mat_data['dYV'].T, dtype=torch.float32)  # Shape [10000, 10]


# Estimate λ before training
lam = max( 10.0 , 5*estimate_lipschitz(x, y) )
print(f"Estimated Lipschitz constant (empirical): {lam:.4f}")





# Compute mean and std for normalization
y_mean = y.mean(dim=0, keepdim=True)  # Shape [1, 10]
y_std = y.std(dim=0, keepdim=True) + 1e-8  # Avoid division by zero

# Normalize y
y_norm = (y - y_mean) / y_std  # Shape [10000, 10]

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create DataLoader for mini-batch training
batch_size = 20
dataset = TensorDataset(x, y_norm)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define Neural Network Model with Two Hidden Layers
class ReLUNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(ReLUNetwork, self).__init__()
        self.hidden1 = nn.Linear(input_dim, hidden_dim1)
        self.relu = nn.ReLU()
        self.hidden2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.output = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        x = self.hidden1(x)
        x = self.relu(x)
        x = self.hidden2(x)
        x = self.relu(x)
        x = self.output(x)
        return x

# Define Model and move to GPU
input_dim   = x.shape[1]
hidden_dim1 = input_dim

output_dim  = y.shape[1]
hidden_dim2 = output_dim
model = ReLUNetwork(input_dim, hidden_dim1, hidden_dim2, output_dim).to(device)

# Define Loss and Optimizer
criterion = nn.MSELoss(reduction='sum')
optimizer = optim.Adam(model.parameters(), lr=0.01)


# Training Loop
epochs = 500

start_time =time.time()

for epoch in range(epochs):
    total_loss = 0
    for x_batch, y_batch in dataloader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if epoch % 10 == 0:
        print(f'Epoch [{epoch}/{epochs}], Loss: {total_loss / len(dataloader):.4f}')
        
        
    if (epoch % 10 == 0)  and (epoch > 0.7*epochs):
        # === Enforce Lipschitz constraint per layer ===
        with torch.no_grad():
            for layer in [model.hidden1, model.hidden2, model.output]:
                weight = layer.weight.data
                norm = torch.linalg.norm(weight, ord=2)
                scale = max(1.0, norm.item() / lam)
                print(f'The scale is calculated as [{scale}]')
                layer.weight.data = weight / scale



# Extract trained parameters for Two hidden layer model
W1 = model.hidden1.weight.detach().cpu().numpy()
b1 = model.hidden1.bias.detach().cpu().numpy()
W2 = model.hidden2.weight.detach().cpu().numpy()
b2 = model.hidden2.bias.detach().cpu().numpy()
W3 = model.output.weight.detach().cpu().numpy()
b3 = model.output.bias.detach().cpu().numpy()

# Reshape y_std to [10, 1] for correct broadcasting
y_std_np = y_std.numpy().reshape(-1, 1)

# Denormalize final layer (W3, b3)
W3_denorm = (y_std_np * W3)
b3_denorm = (y_std.numpy() * b3) + y_mean.numpy()


end_time = time.time()

training_time = end_time - start_time

# Save to MATLAB format
save_path = r"/home/hashemn/Documents/Navid/M2NIST_project/Automatic_1/trained_relu_weights_2h_norm.mat"
scipy.io.savemat(save_path, {
    'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3': W3_denorm, 'b3': b3_denorm, 'Model_training_time': training_time
})

print(f"Trained parameters saved to {save_path}")























# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# import scipy.io
# import numpy as np

# # Define the path to the .mat file
# mat_file_path = r"/home/verivital/Navid/CamVid_project_de_0001/BiSeNet_Reach_eps_1/Train_data2.mat"

# # Load MATLAB data
# mat_data = scipy.io.loadmat(mat_file_path)

# # Extract and transpose data for PyTorch
# x = torch.tensor(mat_data['X'].T, dtype=torch.float32)  # Shape [10000, 5376]
# y = torch.tensor(mat_data['dYV'].T, dtype=torch.float32)  # Shape [10000, 10]

# # Compute mean and std for normalization
# y_mean = y.mean(dim=0, keepdim=True)  # Shape [1, 10]
# y_std = y.std(dim=0, keepdim=True) + 1e-8  # Avoid division by zero

# # Normalize y
# y_norm = (y - y_mean) / y_std  # Shape [10000, 10]

# # Check for GPU availability
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# # Create DataLoader for mini-batch training
# batch_size = 20
# dataset = TensorDataset(x, y_norm)  # Use normalized y
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# # # Define Neural Network Model with One Hidden Layer
# # class ReLUNetwork(nn.Module):
# #     def __init__(self, input_dim, hidden_dim, output_dim):
# #         super(ReLUNetwork, self).__init__()
# #         self.hidden = nn.Linear(input_dim, hidden_dim)  # Hidden Layer
# #         self.relu = nn.ReLU()  # ReLU Activation
# #         self.output = nn.Linear(hidden_dim, output_dim)  # Output Layer
# # 
# #     def forward(self, x):
# #         x = self.hidden(x)
# #         x = self.relu(x)
# #         x = self.output(x)
# #         return x
    
# # Define Neural Network Model with Two Hidden Layer
# class ReLUNetwork(nn.Module):
#     def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
#         super(ReLUNetwork, self).__init__()
#         self.hidden1 = nn.Linear(input_dim, hidden_dim1)  # Hidden Layer
#         self.relu = nn.ReLU()  # ReLU Activation
#         self.hidden2 = nn.Linear(hidden_dim1, hidden_dim2)  # Output Layer
#         self.relu = nn.ReLU()  # ReLU Activation
#         self.output = nn.Linear(hidden_dim2, output_dim)  # Output Layer

#     def forward(self, x):
#         x = self.hidden1(x)
#         x = self.relu(x)
#         x = self.hidden2(x)
#         x = self.relu(x)
#         x = self.output(x)
#         return x

# # # Define Model and move to GPU for One hidden layer
# # input_dim  = 1  # 5376
# # hidden_dim = 60  # One hidden layer with 20 neurons
# # output_dim = 120  # Output dimension
# # model = ReLUNetwork(input_dim, hidden_dim, output_dim).to(device)

# # Define Model and move to GPU for Two hidden layer
# input_dim   = 102  # 5376
# hidden_dim1 = 100  # One hidden layer with 20 neurons
# hidden_dim2 = 100
# output_dim  = 120  # Output dimension
# model = ReLUNetwork(input_dim, hidden_dim1, hidden_dim2, output_dim).to(device)

# # Define Loss and Optimizer
# # criterion = nn.MSELoss()
# criterion = nn.MSELoss(reduction = 'sum')
# # criterion = nn.L1Loss(reduction = 'sum')
# optimizer = optim.Adam(model.parameters(), lr=0.01)

# # Training Loop
# epochs = 50000
# for epoch in range(epochs):
#     total_loss = 0
#     for x_batch, y_batch in dataloader:
#         x_batch, y_batch = x_batch.to(device), y_batch.to(device)  # Move to GPU

#         optimizer.zero_grad()

#         # Forward Pass
#         y_pred = model(x_batch)  # Shape [batch_size, 10]

#         # Compute Loss
#         loss = criterion(y_pred, y_batch)  # Both are [batch_size, 10]

#         # Backward Pass
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()

#     # Print average loss per epoch
#     if epoch % 10 == 0:
#         print(f'Epoch [{epoch}/{epochs}], Loss: {total_loss / len(dataloader):.4f}')

# # # Extract trained parameters for One hidden layer model
# # W1 = model.hidden.weight.detach().cpu().numpy()  # Shape [20, 5376]
# # b1 = model.hidden.bias.detach().cpu().numpy()  # Shape [20]
# # W2 = model.output.weight.detach().cpu().numpy()  # Shape [10, 20]
# # b2 = model.output.bias.detach().cpu().numpy()  # Shape [10]



# # Extract trained parameters for Two hidden layer model
# W1 = model.hidden1.weight.detach().cpu().numpy()  # Shape [20, 5376]
# b1 = model.hidden1.bias.detach().cpu().numpy()  # Shape [20]
# W2 = model.hidden2.weight.detach().cpu().numpy()  # Shape [20, 5376]
# b2 = model.hidden2.bias.detach().cpu().numpy()  # Shape [20]
# W3 = model.output.weight.detach().cpu().numpy()  # Shape [10, 20]
# b3 = model.output.bias.detach().cpu().numpy()  # Shape [10]




# # Reshape y_std to [10, 1] for correct broadcasting
# y_std_np = y_std.numpy().reshape(-1, 1)  # Shape [10, 1]

# # # Denormalize weights and bias for one hidden layer model
# # W2_denorm = (y_std_np * W2)  # Shape [10, 20]
# # b2_denorm = (y_std.numpy() * b2) + y_mean.numpy()  # Shape [10]

# # Denormalize weights and bias for two hidden layer model
# W3_denorm = (y_std_np * W3)  # Shape [10, 20]
# b3_denorm = (y_std.numpy() * b3) + y_mean.numpy()  # Shape [10]

# # # Save to MATLAB format for One hidden Layer model
# # save_path = r"/home/verivital/Navid/CamVid_project/BiSeNet_Reach_eps_1/trained_relu_weights_1h_norm.mat"
# # scipy.io.savemat(save_path, {
# #     'W1': W1, 'b1': b1, 'W2': W2_denorm, 'b2': b2_denorm
# # })


# # Save to MATLAB format for Two hidden Layer model
# save_path = r"/home/verivital/Navid/CamVid_project_de_0001/BiSeNet_Reach_eps_1/trained_relu_weights_2h_norm.mat"
# scipy.io.savemat(save_path, {
#     'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3': W3_denorm, 'b3': b3_denorm
# })

# print(f"Trained parameters saved to {save_path}")