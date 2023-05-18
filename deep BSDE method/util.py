import torch
import torch.nn as nn
from collections import OrderedDict

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim = 20, layer_num = 2, batch_norm = False):
        super(MLP,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.layer_num = layer_num
        self.batch_norm = batch_norm
        if layer_num == 0:
            self.mlp = nn.Linear(input_dim, output_dim)
        else:
            if batch_norm:
                self.mlp = nn.Sequential(
                    OrderedDict(
                    [("hidden_layer_0",
                        nn.Sequential(
                            nn.Linear(input_dim, hidden_dim),
                            nn.BatchNorm1d(hidden_dim),
                            nn.Tanh()
                        )
                      )] + 
                    [("hidden_layer_"+str(i+1),
                        nn.Sequential(
                            nn.Linear(hidden_dim,hidden_dim),
                            nn.BatchNorm1d(hidden_dim),
                            nn.Tanh()
                        )) for i in range(layer_num-1)] + 
                    [("output_layer",nn.Linear(hidden_dim, output_dim))]
                    )
                )
            else:
                self.mlp = nn.Sequential(
                    OrderedDict(
                    [("hidden_layer_0",
                        nn.Sequential(
                            nn.Linear(input_dim, hidden_dim),
                            nn.Tanh()
                        )
                      )] + 
                    [("hidden_layer_"+str(i+1),
                        nn.Sequential(
                            nn.Linear(hidden_dim,hidden_dim),
                            nn.Tanh()
                        )) for i in range(layer_num-1)] + 
                    [("output_layer",nn.Linear(hidden_dim, output_dim))]
                    )
                )
    
    def forward(self, input):
        return self.mlp(input)
    

class ResLayer(nn.Module):
    def __init__(self, dim, hidden_dim = 20, layer_num = 1) -> None:
        super(ResLayer,self).__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.layer_num = layer_num
        self.mlp = MLP(dim,dim,hidden_dim,layer_num)

    def forward(self, inputs):
        return inputs + self.mlp(inputs)


class ResMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim = 20, layer_num = 2) -> None:
        super(ResMLP,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.layer_num = layer_num
        if layer_num == 0:
            self.res_mlp = nn.Linear(input_dim,output_dim)
        else:
            self.res_mlp = nn.Sequential(
                    OrderedDict(
                    [("input_layer",
                        nn.Linear(input_dim,hidden_dim)
                      )] + 
                    [("hidden_layer_"+str(i),
                        ResLayer(hidden_dim,hidden_dim*2,1)
                     ) for i in range(layer_num)] + 
                    [("output_layer",nn.Linear(hidden_dim, output_dim))]
                    )
                )
        self.apply(self._init_weights)
            
    def forward(self, inputs):
        return self.res_mlp(inputs)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0, std=0.05)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    

class ResGrad(ResMLP):
    def __init__(self, input_dim, output_dim, hidden_dim=20, layer_num=2) -> None:
        super().__init__(input_dim+1, output_dim, hidden_dim, layer_num)

    def forward(self, n, x):
        return super().forward(torch.cat([n,x],dim=1))
    

class MLPComplex(MLP):
    def __init__(self, input_dim, output_dim, hidden_dim = 20, layer_num=2, batch_norm = False):
        super(MLPComplex,self).__init__(input_dim, output_dim*2, hidden_dim, layer_num, batch_norm)
        self.output_dim = output_dim

    def forward(self, input):
        output = super().forward(input)
        return output[:,:self.output_dim] + 1j*output[:,self.output_dim:]
    

class MLPResult(MLP):
    def __init__(self, input_dim, output_dim, hidden_dim=20, layer_num=2, batch_norm=False):
        super(MLPResult,self).__init__(input_dim, output_dim, hidden_dim, layer_num, batch_norm)
    
    def forward(self, n, x):
        return super().forward(x)


class MLPGrad(MLP):
    def __init__(self, input_dim, output_dim, hidden_dim=20, layer_num=2, batch_norm=False):
        super(MLPGrad,self).__init__(input_dim+1, output_dim, hidden_dim, layer_num, batch_norm)
    
    def forward(self, n, x):
        return super().forward(torch.cat([n,x],dim=1))
    

class MLPResultComplex(MLPComplex):
    def __init__(self, input_dim, output_dim, hidden_dim=20, layer_num=2, batch_norm=False):
        super(MLPResultComplex,self).__init__(input_dim, output_dim, hidden_dim, layer_num, batch_norm)
    
    def forward(self, n, x, p):
        return super().forward(p)

    
class MLPGradComplex(MLPComplex):
    def __init__(self, input_dim, output_dim, hidden_dim=20, layer_num=2, batch_norm=False):
        super(MLPGradComplex,self).__init__(input_dim+2, output_dim, hidden_dim, layer_num, batch_norm)

    def forward(self, n, x, p):
        return super().forward(torch.cat([n,x,p],dim=1))


class ParameterResult(nn.Module):
    def __init__(self, dim, value=0.5, sd=0.05) -> None:
        super(ParameterResult,self).__init__()
        self.dim = dim
        self.result = nn.Parameter(data=torch.randn(dim)*sd+value,requires_grad=True)
    
    def forward(self, n, x):
        batch_size = n.shape[0]
        return torch.ones([batch_size,self.dim])*self.result