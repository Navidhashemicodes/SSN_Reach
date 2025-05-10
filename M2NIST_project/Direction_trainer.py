import torch
import numpy as np
import scipy.io

# Define the function to compute covariance
def compute_f(A, X_batch):
    """
    Compute the function f(A) = (1 / (N * norm(A)^2)) * sum_i (A' * (X_i - mean(X)))^2
    """
    norm_A2 = torch.norm(A, p=2)
    mean_X = torch.mean(X_batch, dim=1, keepdim=True)  
    N = X_batch.shape[1]
    dX = X_batch - mean_X
    A_X = torch.matmul(A.T, dX)  # A' * (X - mean)
    
    # Compute covariance
    cov_batch = (1 / (norm_A2)) * torch.sqrt( torch.sum(torch.pow(A_X, 2)) / N )
    return cov_batch


# Define the path to the .mat file
mat_file_path = r"C:\Users\navid\Documents\MATLAB\MATLAB_prev\others\Files\CDC2023\Large_DNN\Case_study\CAV_21_New_reach\Train_data.mat"

# Load MATLAB data
mat_data = scipy.io.loadmat(mat_file_path)  

# Extract individual arrays
Y1 = torch.tensor(mat_data['Y1'], dtype=torch.float32)
Y2 = torch.tensor(mat_data['Y2'], dtype=torch.float32)
Y3 = torch.tensor(mat_data['Y3'], dtype=torch.float32)

# Reshape each array to (64*84*11, 2000)
new_shape = (64 * 84 * 11, 2000)
Y1 = Y1.reshape(new_shape)
Y2 = Y2.reshape(new_shape)
Y3 = Y3.reshape(new_shape)

# Concatenate along the second axis to get shape (64*84*11, 5*2000)
X = torch.cat((Y1, Y2, Y3), dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Send the tensor X to GPU
X = X.to(device)

# Print the final shape
print(X.shape)  # Should be (64*84*11, 5*2000)


# Initialize the input data X and parameters A
n = X.shape[0]
m = X.shape[1]  # Number of data points

# Set hyperparameters

batch_size = 512  # Batch size for training


# Initialize Cov_prev with a very high value to start
Cov_prev = float('inf')

# Function to perform the entire deflation process
def deflation(X, n, m, device):
    A_list = []  # To store all principal directions
    round = -1
    while True:
        round += 1
        # Initialize A for this round
        
        A = torch.randn(n, 1, requires_grad=True, device=device)  # Trainable parameter (n x 1)
        
        Initial_lr = n
        
        optimizer = torch.optim.SGD([A], lr=Initial_lr)
        
        Cov_prev = float('inf')
        iter = 0
        
        # Train A to find the principal direction for this round
        while True:
            optimizer.zero_grad()  # Zero gradients
            
            
            # Sample a random batch from X
            indices = torch.randint(0, m, (batch_size,), device=device)  
            X_batch = X[:, indices]  # Select batch
            
            # Compute the function value f(A) and its gradients
            cov_batch = compute_f(A, X_batch)
            
            # Perform gradient ascent to maximize cov_batch
            (-cov_batch).backward(retain_graph=True)  # Equivalent to maximizing cov_batch
            
            optimizer.step()  # Update A
            
            
            # Check convergence
            if iter % 100 == 0:
                Cov = compute_f(A, X)
                print(f"Round {round+1} - Iteration {iter} - Covariance: {Cov.item()}")
                
                # Check if convergence condition is met
                if abs(Cov - Cov_prev) < Cov_prev * 1e-3:
                    print(f"Convergence achieved at iteration {iter}. Stopping optimization.")
                    break
                Cov_prev = Cov  # Update the previous covariance value
            
            iter += 1
        
        # Store the current principal direction A
        A = A / torch.norm( A , p=2 )
        A_list.append(A.clone().detach())
        
        # Remove the component of the data in the direction of A
        X_proj = torch.matmul(A.T, X) * A  # Project the data onto A
        del A
        print(X_proj.shape)  # Should be (64*84*11, 5*2000)
        X = X - X_proj  # Subtract projection from X to remove the direction
        X = X.detach()
        
        
        if round == 0:
            Largest = Cov
        
        
        if Cov<=Largest/20:
             print(f"A sufficient amount of principal directions ( {round+1} unit vectors) is collected.")
             break
         
    return A_list, X



# Perform deflation
A_list, X_updated = deflation(X, n, m, device)

# Print the final updated X and principal directions
print("Updated X shape:", X_updated.shape)
print("Principal directions found:")
for i, A in enumerate(A_list):
    print(f"A{i+1}: {A}")
    

A_list = np.hstack([A.to('cpu').numpy().astype(np.float32) for A in A_list])
scipy.io.savemat('directions.mat', {'Directions': A_list})
    