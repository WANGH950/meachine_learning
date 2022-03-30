import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP,self).__init__()
        self.mlp_seq = nn.Sequential(
            nn.Linear(input_dim,input_dim+10),
            nn.Sigmoid(),
            nn.Linear(input_dim+10,input_dim+10),
            nn.Sigmoid(),
            nn.Linear(input_dim+10,output_dim)
        )
    
    def forward(self,input_tensor):
        return self.mlp_seq(input_tensor)

class MLP_v1(nn.Module):
    def __init__(self, input_dim, output_dim, kmin, kmax):
        super(MLP_v1,self).__init__()
        self.kmin = kmin
        self.kmax = kmax
        self.mlp_seq = nn.Sequential(
            nn.Linear(input_dim,output_dim*2),
            nn.Tanh(),
            nn.Linear(output_dim*2,output_dim*4),
            nn.Tanh(),
            nn.Linear(output_dim*4,output_dim*2),
            nn.Tanh(),
            nn.Linear(output_dim*2,output_dim)
        )
        self.relu = nn.ReLU()
    
    def forward(self,input_tensor):
        limits = torch.zeros([input_tensor.shape[0],1])
        for i in range(input_tensor.shape[0]):
            if input_tensor[i,1] > 0:
                limits[i,0] = self.relu(input_tensor[i,0] - self.kmin*input_tensor[i,1])*self.relu(self.kmax*input_tensor[i,1] - input_tensor[i,0])/(input_tensor[i,1]*self.kmin)**2
        self.mlp_outputs = self.mlp_seq(input_tensor)
        output_tensor = self.mlp_outputs**2*limits.detach()
        return output_tensor