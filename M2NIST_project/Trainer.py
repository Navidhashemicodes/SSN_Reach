import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import scipy.io

# Define the path to the .mat file
mat_file_path = r"C:\Users\navid\Documents\MATLAB\MATLAB_prev\others\Files\CDC2023\Large_DNN\Case_study\CAV_21_New_reach\Train_data.mat"

# Load MATLAB data
mat_data = scipy.io.loadmat(mat_file_path)

# Extract individual arrays and transpose them
x = torch.tensor(mat_data['X'].T, dtype=torch.float32)  # Shape [10000, 5376]
y = torch.tensor(mat_data['dYV'].T, dtype=torch.float32)  # Shape [10000, 10]

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create a dataset and DataLoader for mini-batch training
batch_size = 100  # Adjust based on your GPU memory
dataset = TensorDataset(x, y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define Linear Model
class LinearModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)  # y = W*x + b

    def forward(self, x):
        return self.linear(x)

# Define Model and move to GPU
input_dim = 1  # 5376
output_dim = 10  # Corrected to match `dYV` dimensions
model = LinearModel(input_dim, output_dim).to(device)

# Define Loss and Optimizer
criterion = nn.MSELoss(reduction = 'sum')
# criterion = nn.L1Loss(reduction = 'sum')
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop with Mini-Batches
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


# Extract and Save Optimized Weights and Biases
W = model.linear.weight.detach().cpu().numpy()  # Shape [10, 5376]
b = model.linear.bias.detach().cpu().numpy()  # Shape [10]

# Save to MATLAB format
save_path = r"C:\Users\navid\Documents\MATLAB\MATLAB_prev\others\Files\CDC2023\Large_DNN\Case_study\CAV_21\trained_weights.mat"
scipy.io.savemat(save_path, {'W': W, 'b': b})

print(f"Trained parameters saved to {save_path}")