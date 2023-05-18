import torch
import numpy as np

def lamb_bkwd(n):
    return (50-n)/10
def mu_bkwd(n):
    result = n/5
    result[n==0] = 0
    return result

def lamb_fwd(n):
    return mu_bkwd(n+1)
def mu_fwd(n):
    result = lamb_bkwd(n-1)
    result[n==0] = 0
    return result

# functional
def U_positional_func(x):
    return torch.norm(x,dim=2,keepdim=True)**2
    
def U_occupation_time(x):
    return torch.relu(torch.sign(x[:,:,:1]))

def g_fwd(n, x):
    dim = x.shape[1]
    u_rel = torch.exp(-torch.norm(x,dim=1,keepdim=True)**2)/torch.sqrt(torch.tensor(np.pi)**dim)*torch.relu(torch.sign(10-n))/10
    return u_rel + 1j*0

def f_fwd(t,n,x,p,u,grad):
    param = lamb_fwd(n) + mu_fwd(n) - lamb_fwd(n-1) - mu_fwd(n+1)
    return param*u

def g_bkwd(n, x):
    return torch.ones_like(n) + 1j*0

def f_bkwd(t,n,x,p,u,grad):
    return 0