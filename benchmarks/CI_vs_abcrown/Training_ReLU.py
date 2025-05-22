import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import scipy.io
import numpy as np
import random
from time import time

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


def Trainer_ReLU(x , y , device, epochs, save_path):
    
    x = x.T.to(device)
    y = y.T.to(device)
    
    # Estimate λ before training
    lam = max( 10.0 , 5*estimate_lipschitz(x, y) )
    print(f"Estimated Lipschitz constant (empirical): {lam:.4f}")


    # Compute mean and std for normalization
    y_mean = y.mean(dim=0, keepdim=True)  # Shape [1, 10]
    y_std = y.std(dim=0, keepdim=True) + 1e-8  # Avoid division by zero

    # Normalize y
    y_norm = (y - y_mean) / y_std  # Shape [10000, 10]

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
    output_dim  = y.shape[1]
    input_dim   = x.shape[1]
    hidden_dim1 = output_dim
    hidden_dim2 = output_dim
    model = ReLUNetwork(input_dim, hidden_dim1, hidden_dim2, output_dim).to(device)

    # Define Loss and Optimizer
    criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=0.01)


    # Training Loop
    t0 = time()
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
    y_std_np = y_std.cpu().numpy().reshape(-1, 1)

    # Denormalize final layer (W3, b3)
    W3_denorm = (y_std_np * W3)
    b3_denorm = (y_std.cpu().numpy() * b3) + y_mean.cpu().numpy()

    # Save to MATLAB format
    scipy.io.savemat(save_path, {
        'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3': W3_denorm, 'b3': b3_denorm
    })

    print(f"Trained parameters saved to {save_path}")
    
    model.output.weight.data = torch.tensor(W3_denorm).to(device)
    model.output.bias.data = torch.tensor(b3_denorm).to(device)
    
    train_time = time() - t0
    
    return model , train_time