import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import scipy.io
import numpy as np

# Define the path to the .mat file
mat_file_path = r"C:\Users\navid\Documents\MATLAB\MATLAB_prev\others\Files\CDC2023\Large_DNN\Case_study\CAV_21\Train_data.mat"

# Load MATLAB data
mat_data = scipy.io.loadmat(mat_file_path)

# Extract and transpose data for PyTorch
x = torch.tensor(mat_data['X'].T, dtype=torch.float32)  # Shape [10000, 5376]
y = torch.tensor(mat_data['dYV'].T, dtype=torch.float32)  # Shape [10000, 10]

# Compute mean and std for normalization
y_mean = y.mean(dim=0, keepdim=True)  # Shape [1, 10]
y_std = y.std(dim=0, keepdim=True) + 1e-8  # Avoid division by zero

# Normalize y
y_norm = (y - y_mean) / y_std  # Shape [10000, 10]

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create DataLoader for mini-batch training
batch_size = 100
dataset = TensorDataset(x, y_norm)  # Use normalized y
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define Neural Network Model with One Hidden Layer
class ReLUNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ReLUNetwork, self).__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)  # Hidden Layer
        self.relu = nn.ReLU()  # ReLU Activation
        self.output = nn.Linear(hidden_dim, output_dim)  # Output Layer

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        return x

# Define Model and move to GPU
input_dim  = 1  # 5376
hidden_dim = 10  # One hidden layer with 20 neurons
output_dim = 10  # Output dimension
model = ReLUNetwork(input_dim, hidden_dim, output_dim).to(device)

# Define Loss and Optimizer
# criterion = nn.MSELoss()
criterion = nn.MSELoss(reduction = 'sum')
# criterion = nn.L1Loss(reduction = 'sum')
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
epochs = 500
for epoch in range(epochs):
    total_loss = 0
    for x_batch, y_batch in dataloader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)  # Move to GPU

        optimizer.zero_grad()

        # Forward Pass
        y_pred = model(x_batch)  # Shape [batch_size, 10]

        # Compute Loss
        loss = criterion(y_pred, y_batch)  # Both are [batch_size, 10]

        # Backward Pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Print average loss per epoch
    if epoch % 10 == 0:
        print(f'Epoch [{epoch}/{epochs}], Loss: {total_loss / len(dataloader):.4f}')

# Extract trained parameters
W1 = model.hidden.weight.detach().cpu().numpy()  # Shape [20, 5376]
b1 = model.hidden.bias.detach().cpu().numpy()  # Shape [20]
W2 = model.output.weight.detach().cpu().numpy()  # Shape [10, 20]
b2 = model.output.bias.detach().cpu().numpy()  # Shape [10]

# Reshape y_std to [10, 1] for correct broadcasting
y_std_np = y_std.numpy().reshape(-1, 1)  # Shape [10, 1]

# Denormalize weights and bias
W2_denorm = (y_std_np * W2)  # Shape [10, 20]
b2_denorm = (y_std.numpy() * b2) + y_mean.numpy()  # Shape [10]

# Save to MATLAB format
save_path = r"C:\Users\navid\Documents\MATLAB\MATLAB_prev\others\Files\CDC2023\Large_DNN\Case_study\CAV_21\trained_relu_weights_norm.mat"
scipy.io.savemat(save_path, {
    'W1': W1, 'b1': b1, 'W2': W2_denorm, 'b2': b2_denorm
})

print(f"Trained parameters saved to {save_path}")