import os
import numpy as np
from tqdm import tqdm  # Import tqdm for the progress bar
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# Set the number of threads for PyTorch
torch.set_num_threads(2)

# Define a neural network block with residual connections 
class DeepRitzBlock(nn.Module): #M: (new class that inherits from the nn class)
    def __init__(self, h_size):
        super(DeepRitzBlock, self).__init__()
        # Define a sequential block with two linear layers and Tanh activations
        self.block = nn.Sequential(
            nn.Linear(h_size, h_size),  #M: h_size is size of input- and output-samples here
            nn.Tanh(),
            nn.Linear(h_size, h_size),
            nn.Tanh()
        )

    def forward(self, x):
        # Apply the block and add the input (residual connection) 
        return self.block(x) + x

# Define a neural network model using DeepRitzBlock 
class NeuralNetwork(nn.Module): #M: (new class that inherits from the nn class)
    def __init__(self, in_size, h_size=10, block_size=1, dev="cpu"):
        super(NeuralNetwork, self).__init__()
        self.dev = dev
        self.dim_input = in_size
        self.dim_h = h_size
        self.num_blocks = block_size

        # Initialize the layers list with either a linear layer or padding
        layers = [nn.ConstantPad1d((0, self.dim_h - self.dim_input), 0) if self.dim_input <= self.dim_h else
                  nn.Linear(self.dim_input, self.dim_h)]                       #TODO: does linear layer represent d(x)?
                    #M: padding extends the 1d input to desired shape, if input shape is smaller or equal;
                    #if input is > desired shape, a linear layer projects input into desired dimension 

        # Append num_block DeepRitzBlock-instances to the layers list
        for _ in range(self.num_blocks):
            layers.append(DeepRitzBlock(self.dim_h))

        # Add a final linear layer to map back to the input size
        layers.append(nn.Linear(self.dim_h, self.dim_input))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # Pass the input through the model
        #M: NN(x) = Wfinal​(pad or linear(x)) = d(x); in simplest case, only the final linear layer makes up d(x)
        #M: with deep ritz block, nn always contains space of linear functions as subset of its representational capacity, but allows richer approximations as well
        return self.model(x)

def generate_surrogate(x): #M: this function is never called in this script
    x_surrogate = []
    for i, x_i in enumerate(x):
        x_surrogate_ = []
        for l in range(len(x_i)):
            fft_original = np.fft.rfft(x_i[l])
            magnitude = np.abs(fft_original)
            random_phases = np.exp(1j * np.random.uniform(0, 2 * np.pi, len(fft_original)))
            fft_random = magnitude * random_phases
            random_signal = np.fft.irfft(fft_random)
            x_surrogate_.append(random_signal)
        x_surrogate.append(np.array(x_surrogate_))
    return x_surrogate

def EPR_nn_est_params(data_, dt, epoch_max=3000, step_test=100, data_normalisation=True,
                      reverse_training=False, dim_h=10, num_blocks=2):
    '''M: Estimates EPR of time series
    ------------ parameters -------------
     - ...
     - 
     - step_test: determines how many intermediate loss should be logged in all_losses
     - reverse_training: switches, which half of input to use for training and which for testing
     - dim_h: minimum size of network layers (what does it change?), if input-dim is smaller, input is padded
     - num_blocks: number of Deep-Ritz-Block layers in the NN (non-linear additions to input for flexibility)'''
    data_ij = data_ 
    dim, length = data_ij.shape #M: get dimensionality and duration of input

    if reverse_training:
        #M: use 2nd half of data for training and 1st for testing
        xt = torch.Tensor(data_ij[:, length // 2:].T)
        xt_test = torch.Tensor(data_ij[:, :length // 2].T)
    else:
        xt = torch.Tensor(data_ij[:, :length // 2].T) #M: xt is train data, tensor of shape (length, dim), e.g. (time, neuron)
        xt_test = torch.Tensor(data_ij[:, length // 2:].T)

    dim_x = dim
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu") #M: TODO: what happens here?
    force_network = NeuralNetwork(dim_x, dim_h, num_blocks, dev).to(dev) #M: Create a NeuralNetwork 

    #M: standardize train data
    data = xt - torch.mean(xt, dim=0)
    if data_normalisation:
        data /= torch.std(data, dim=0)

    #M: data_mid and xdiff are for computation of current; see eq. A1 in Elias' report
    data_mid = 0.5 * (data[1:, :] + data[:-1, :]) 
    xdiff = data[1:, :] - data[:-1, :]

    optimizer = torch.optim.Adadelta(force_network.parameters(), lr=1e-2) #M: will optimize network to find optimal...

    #M: start training
    All_loss_train = []
    print('training') #M
    for epoch in tqdm(range(epoch_max), total=epoch_max): #M (tqdm)
        force_network.train() #M: set network to train mode
        optimizer.zero_grad() 
        dxmid = force_network(data_mid) #M: force_network optimizes loss w.r.t function d(data_mid); uses
        jj = torch.sum(dxmid * xdiff, dim=1) #M: get current 'jj' for all dimensions together (sum across j's of neurons for each tp) with estimated function in dxmid; see eq. A1 in Elias' report
        loss = -2 * torch.mean(jj) ** 2 / (dt * torch.var(jj)) #M: loss-func minimization (maximize lower bound 'loss' w.r.t. current 'jj');
        #M: see eq. 2.23 in Elias' report
        loss.backward() #M: calculate gradient and backpropagate, such that force_network calculates more optimal d(x) in next epoch TODO correct?
        optimizer.step()

        if epoch % step_test == 0:
            torch.save(force_network.state_dict(), f'saved_params/force_network_params_epoch_{epoch}.pt') #M: save learnable parameters
            All_loss_train.append(-loss.item())

    saved_params_dir = 'saved_params'
    steps = tqdm(range(0, epoch_max, step_test), total=epoch_max/step_test) #M (tqdm)
    test_losses = []

    data_test = xt_test - torch.mean(xt_test, dim=0)
    if data_normalisation:
        data_test /= torch.std(data_test, dim=0)

    data_mid_test = 0.5 * (data_test[1:, :] + data_test[:-1, :])
    xdiff_test = data_test[1:, :] - data_test[:-1, :]
    print('training') #M
    for step in steps: 
        #M: load saved params from trained network
        param_file = os.path.join(saved_params_dir, f'force_network_params_epoch_{step}.pt') 
        if os.path.exists(param_file):
            force_network.load_state_dict(torch.load(param_file))
            force_network.eval() #M: prepare
            with torch.no_grad():
                dxmid_test = force_network(data_mid_test) #M: force_network is my function d(x)
                jj_test = torch.sum(dxmid_test * xdiff_test, dim=1)
                loss_test = -2 * torch.mean(jj_test) ** 2 / (dt * torch.var(jj_test))
                #M: Note that, unlike for training, loss.backward() isn't called here, as no backpropagation is desired during testing
                test_losses.append(-loss_test.item())
        else:
            test_losses.append(None)

    epr_train, epr_test = np.max(All_loss_train), np.max(test_losses)
    print(f"Maximum EPR on training data: {epr_train}")
    print(f"Maximum EPR on test data: {epr_test}")

    return test_losses, All_loss_train

# Test with random a timeseries of dimension N timelength T
T = 100000  # Length of the time series
N = 10    # Number of dimensions

# Generate random time series data
time_series = np.random.randn(N, T)

# Change the current working directory: IN WHICH YOU HAVE A FOLDER NAMED "saved_params"
os.chdir('/Users/michi/Documents/2 Neuro/Praktikum KTH/Code') #Users/pascal.helson/Documents/PYTHON/testing_code')#

# Verify the change
print("Current Working Directory:", os.getcwd())

# Setting parameters
epoch_max, step_test, dim_h, num_blocks = 500, 10, 5, 1

test_losses, All_loss_train = EPR_nn_est_params(time_series, dt=0.005,
                    epoch_max=epoch_max, step_test=step_test, reverse_training=False, dim_h=dim_h, num_blocks=num_blocks)

sigma_test = test_losses
sigma_train = All_loss_train
test_array = np.arange(0,epoch_max,1)
plt.plot(test_array[::step_test],sigma_test, label = 'Test')
train_array = np.arange(0,epoch_max,1)
plt.plot(train_array[::step_test], sigma_train[::1], label = 'Train')
#plt.axhline(epr, color = 'r')
plt.xlabel('Epoch')
plt.ylabel('Entropy Production Rate')
plt.legend()
plt.show()
