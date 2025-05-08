from trainer_original import *
from functions import export2matlab


import torch

import torch.nn as nn

import torch.optim as optim

import numpy as np

from torch.utils.data import TensorDataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

from scipy.io import loadmat



def main():

    print(2020)
    train_batch_size = 2
    file_path = 'C:/Users/navid/Documents/MATLAB/MATLAB_prev/others/Files/CDC2023/Large_DNN/Case_study/ModelandData2.mat'
    
    data = loadmat(file_path)

    Xtrain = data['X_n']
    Xtrain = torch.Tensor(np.transpose(Xtrain[:,:5000]))
    Ytrain = data['Y_n']
    Ytrain = torch.Tensor(np.transpose(Ytrain[:,:5000]))
    print(Xtrain.shape)
    print(Ytrain.shape)
    
    
    trainset=TensorDataset(Xtrain, Ytrain)
    trainloader = DataLoader(trainset, batch_size= train_batch_size, shuffle=True, num_workers=6)
    
    print('Train data loaded')


    # ############### Initialize if you want !! ###############################
    # initweights_file_path = 'C:/Users/navid/Documents/MATLAB/MATLAB_prev/others/Files/CDC2023/Large_DNN/src/Lip_bounded_trainer/init_weights.mat'
    # initbiases_file_path = 'C:/Users/navid/Documents/MATLAB/MATLAB_prev/others/Files/CDC2023/Large_DNN/src/Lip_bounded_trainer/init_biases.mat'

    # weights_mat = loadmat(initweights_file_path)['W']
    # biases_mat = loadmat(initbiases_file_path)['b']
            
    # weights = [torch.from_numpy(wi.astype(np.float32)) for wi in weights_mat[0]]
    # biases = [torch.from_numpy(bi.astype(np.float32)) for bi in biases_mat[0]]

    # initial_params = {}
    # for i in range(len(weights)):
    #     initial_params[f'{2 * i}.weight'] = weights[i]
    #     BB = biases[i]
    #     initial_params[f'{2 * i}.bias'] = BB.flatten()


    net = nn.Sequential(
        nn.Linear(784,10),
        nn.ReLU(),
        nn.Linear(10,5),
        nn.ReLU(),
        nn.Linear(5,5),
        nn.ReLU(),
        nn.Linear(5,5),
        nn.ReLU(),
        nn.Linear(5,2)
    )
    
    # net.load_state_dict(initial_params)
    
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    verbose = False
    epoch = 10
    print('Training started')

    train_lip_bound(trainloader, net, 2, optimizer, epoch, verbose)

    export2matlab('Main_network',net)
    
    return net


if __name__ == '__main__':
    net = main()
