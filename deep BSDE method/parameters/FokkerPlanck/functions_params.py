import torch
import numpy as np

def lamb_bkwd(n):
    return (100-n)/20

def mu_bkwd(n):
    result = torch.ones_like(n)*2
    result[n==0] = 0
    return result

def g_bkwd(n, x):
    dim = x.shape[1]
    u0 = torch.exp(-10*torch.norm(x,dim=1,keepdim=True)**2/(n+1))*torch.sqrt((10/(n+1)/torch.tensor(np.pi))**dim)
    return u0.float()

def g_bkwdc(n, x):
    dim = x.shape[1]
    u0 = (torch.exp(-torch.norm(x-1,dim=1,keepdim=True)**2*(n+1)) + torch.exp(-torch.norm(x+1,dim=1,keepdim=True)**2*(n+1)))*torch.sqrt(((n+1)/torch.tensor(np.pi))**dim)/2
    return u0.float()

def f_bkwd(t,n,x,u,grad):
    return 0


def lamb_fwd(n):
    return mu_bkwd(n+1)

def mu_fwd(n):
    result = lamb_bkwd(n-1)
    result[n==0] = 0
    return result

def g_fwd(n, x):
    dim = x.shape[1]
    g = torch.exp(-torch.norm(x,dim=1,keepdim=True)**2)/torch.sqrt(torch.tensor(np.pi)**dim)/2**torch.relu(n-20)/22
    return g.float()

def f_fwd(t,n,x,u,grad):
    param = lamb_fwd(n) + mu_fwd(n) - lamb_fwd(n-1) - mu_fwd(n+1)
    return param*u

def g_bkwd_high_dim(n, x):
    g = (0.5 + 0.49*torch.sin(torch.sum(x,dim=1,keepdim=True)/(n+1)))
    return g.float()

def f_bkwd_high_dim(t,n,x,u,grad):
    return u**3 - u