import time
import torch
import torch.nn as nn
from equations import Equation

class deepBSDE(nn.Module):
    def __init__(self, equation:Equation, result:nn.Module, grad:nn.ModuleList, model_params:dict) -> None:
        super(deepBSDE,self).__init__()
        self.equation = equation
        self.result = result
        self.grad = grad
        self.n = model_params['n']
        self.x = model_params['x']
        self.p = model_params['p']
        self.t = model_params['t']
        self.T = model_params['T']
        self.N = model_params['N']
    
    def forward(self, batch_size):
        return 0, 0

class ExplicitDeepBSDE(deepBSDE):
    def __init__(self, equation:Equation, result:nn.Module, grad:nn.ModuleList, model_params:dict) -> None:
        super(ExplicitDeepBSDE,self).__init__(equation,result,grad,model_params)

    def forward(self, batch_size):
        delta_t = (self.T - self.t) / self.N
        process_N, process_X, discrete_p, discrete_t, delta_B = self.equation.get_positions(self.n, self.x, self.p, self.t, self.T, self.N, batch_size)
        delta_B = delta_B + 0j
        functional = 1j*discrete_p*self.equation.U(process_X)
        functional = torch.roll(functional,1,0)
        functional[0] = 0
        exp_functional = torch.exp(-torch.cumsum(functional,dim=0)*delta_t)
        u = self.result(process_N[0], process_X[0], discrete_p[0])
        for i in range(self.N):
            grad_u = self.grad[i](process_N[i], process_X[i], discrete_p[i])
            grad_bmm = torch.bmm(grad_u.unsqueeze(1), delta_B[i].unsqueeze(-1)).squeeze(-1)
            f = self.equation.f(discrete_t[i], process_N[i], process_X[i], discrete_p[i], u / exp_functional[i], grad_u)
            u = u - exp_functional[i] * f * delta_t + exp_functional[i] * grad_bmm
        g = exp_functional[self.N] * self.equation.g(process_N[self.N],process_X[self.N])
        return u, g

class ImplicitDeepBSDE(deepBSDE):
    def __init__(self, equation:Equation, result:nn.Module, grad:nn.ModuleList, model_params:dict) -> None:
        super(ImplicitDeepBSDE,self).__init__(equation,result,grad,model_params)

    def forward(self, batch_size):
        delta_t = (self.T - self.t) / self.N
        process_N, process_X, discrete_p, discrete_t, delta_B = self.equation.get_positions(self.n, self.x, self.p, self.t, self.T, self.N, batch_size)
        delta_B = delta_B + 0j
        functional = 1j*discrete_p*self.equation.U(process_X)
        functional = torch.roll(functional,1,0)
        functional[0] = 0
        exp_functional = torch.exp(-torch.cumsum(functional,dim=0)*delta_t)
        g = exp_functional[self.N] * self.equation.g(process_N[self.N],process_X[self.N])
        for i in range(self.N):
            k = self.N-i-1
            grad_u = self.grad[k](process_N[k], process_X[k], discrete_p[k])
            grad_bmm = torch.bmm(grad_u.unsqueeze(1), delta_B[k].unsqueeze(-1)).squeeze(-1)
            f = self.equation.f(discrete_t[k], process_N[k], process_X[k], discrete_p[k], g / exp_functional[k+1], grad_u)
            g = g + exp_functional[k] * f * delta_t - exp_functional[k] * grad_bmm
        u = self.result(process_N[0], process_X[0], discrete_p[0])
        return u, g

class MSELoss(nn.Module):
    def __init__(self) -> None:
        super(MSELoss,self).__init__()
    
    def forward(self, input, target):
        loss = nn.MSELoss()
        return loss(input.real,target.real) + loss(input.imag,target.imag)


def train(model, train_params):
    epoch = train_params['epoch']
    batch_size = train_params['batch_size']
    lr = train_params['learning_rate']

    change_lr = train_params['change_lr']
    lr_change = train_params['lr_change']

    # init criterion
    criterion = MSELoss()

    # init Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)

    loss_values = torch.ones(epoch)
    # start training
    start = time.time()
    for i in range(epoch):
        # change learning rate or not?
        if change_lr and i == int(epoch/2):
            for param_grop in optimizer.param_groups:
                param_grop['lr'] = lr_change
        model.train()
        optimizer.zero_grad()

        u,u0 = model(batch_size)
        loss = criterion(u,u0)
        loss.backward()
        optimizer.step()

        model.eval()
        loss_values[i] = loss.item()

        print('\r%5d/{}|{}{}|{:.2f}s  [Loss: %e]'.format(
            epoch,
            "#"*int((i+1)/epoch*50),
            " "*(50-int((i+1)/epoch*50)),
            time.time() - start) %
            (i+1,
            loss_values[i]), end = ' ', flush=True)
    print("\nTraining has been completed.")
    return loss_values