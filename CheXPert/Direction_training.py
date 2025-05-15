import torch
import numpy as np
import scipy.io
import gc
from time import time

# Define the function to compute covariance
# @torch.jit.script
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



# Function to perform the entire deflation process
def deflation(X, n, m, device, batch_size, Cov_prev):
    A_list = []  # To store all principal directions
    round = -1
    Largest = float('inf')

    while True:
        round += 1
        # Initialize A for this round
    
        if round>0:
            del X_batch
            del X_proj
            gc.collect()
    
        A = torch.randn(n, 1, requires_grad=True, device=device)  # Trainable parameter (n x 1)
    
        Initial_lr = 100*n
    
        optimizer = torch.optim.SGD([A], lr=Initial_lr)
    
        Cov_prev = float('inf')
    
        iter = 0
        Cov = compute_f(A, X)
        print(f"Initial guess for covaiance is: {Cov.item()}")
    
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
                if round == 0:
                    if abs(Cov - Cov_prev) < Cov_prev * 1e-2:
                        print(f"Convergence achieved at iteration {iter}. Stopping optimization.")
                        break
                else:
                    if abs(Cov - Cov_prev) < Largest * 1e-2:
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
    
    
        if Cov<=Largest/100:
            print(f"A sufficient amount of principal directions ( {round+1} unit vectors) is collected.")
            break
        
    return A_list, X



def compute_directions(X , device, batch_size):

    # Send the tensor X to GPU
    X = X.to(device).T

    # Print the final shape
    print(X.shape)  # Should be (64*84*11, 5*2000)


    # Initialize the input data X and parameters A
    n = X.shape[0]
    m = X.shape[1]  # Number of data points

    # Set hyperparameters

    # Initialize Cov_prev with a very high value to start
    Cov_prev = float('inf')


    # Perform deflation
    t0 = time()
    A_list, X_updated = deflation(X, n, m, device, batch_size, Cov_prev)
    Direction_training_time = time() - t0
    # Print the final updated X and principal directions
    print("Updated X shape:", X_updated.shape)
    print("Principal directions found:")
    for i, A in enumerate(A_list):
        print(f"A{i+1}: {A}")
        
    return A_list , Direction_training_time