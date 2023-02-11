import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

# dataset class
class IMDbDataset(Dataset):
    """IMDb dataset."""
    def __init__(self, data:np.array, label:np.array, 
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        self.device = device
        self.data = torch.Tensor(data).to(device=device)
        self.label = torch.Tensor(label).to(device=device)
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

# network class
class SimpleNN(nn.Module):
    def __init__(self, in_shape, learning_rate, lamb, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        super(SimpleNN, self).__init__()
        self.in_shape = in_shape
        self.fc1 = nn.Linear(self.in_shape, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        
        self.learning_rate = learning_rate
        self.cost_func = nn.MSELoss()
        self.lamb = lamb
        self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, weight_decay=self.lamb)
        
        self.device = device
        self.to(device)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                module.bias.data.fill_(0.01)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def gradient_magnitude(self, total=True):
        layer1 = torch.norm(self.fc1.weight.grad, p=2).detach().cpu().numpy().flatten()[0] +\
            torch.norm(self.fc1.weight.grad, p=2).detach().cpu().numpy().flatten()[0]
        layer2 = torch.norm(self.fc2.weight.grad, p=2).detach().cpu().numpy().flatten()[0] +\
            torch.norm(self.fc2.weight.grad, p=2).detach().cpu().numpy().flatten()[0]
        layer3 = torch.norm(self.fc3.weight.grad, p=2).detach().cpu().numpy().flatten()[0] +\
            torch.norm(self.fc3.weight.grad, p=2).detach().cpu().numpy().flatten()[0]
        
        if total:
            return np.sqrt(layer1**2 + layer2**2 + layer3**2)
        else:
            return layer1, layer2, layer3
    
    def train_one_step(self, batch_of_data, their_lables):
        self.zero_grad()
        network_output = self.forward(batch_of_data)
        cost = self.cost_func(network_output, their_lables)
        cost.backward()
        self.optimizer.step()
    
# TODO: W&B. But for now, it doesn't seem necessary.