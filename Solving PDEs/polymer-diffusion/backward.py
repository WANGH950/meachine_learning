import torch
import torch.nn as nn
import sys
import polymerDiffusion as model
sys.path.append("../") 
import common

class PINNs_v1(nn.Module):
    def __init__(self, mu, lam, D0, nmin, nmax, smax, tmin, tmax, alpha, batch_size, int_Ns, int_Nt):
        super(PINNs_v1,self).__init__()
        A = torch.zeros([nmax+1,nmax+1])
        A[0,0] = -lam
        A[0,1] = lam
        A[nmax,nmax] = -mu
        A[nmax,nmax-1] = mu
        for i in range(nmax-1):
            A[i+1,i] = mu
            A[i+1,i+1] = -(mu+lam)
            A[i+1,i+2] = lam
        self.polymer_diffusion = model.Polymer_Diffusion(A,D0,nmin,nmax, smax,tmin, tmax, alpha, int_Ns, int_Nt, batch_size)
        self.mlp = common.MLP_v1(2,nmax+1,self.polymer_diffusion.kmin,self.polymer_diffusion.kmax)

    def forward(self):
        f_samples, int_samples = self.polymer_diffusion.get_samples()
        f_ = self.polymer_diffusion.f(self.mlp,f_samples[:,0:1],f_samples[:,1:2])
        # 积分正则项
        int_ = torch.ones([self.polymer_diffusion.int_Nt,self.polymer_diffusion.nmax+1])
        for i in range(int_.shape[0]):
            results = self.mlp(int_samples[i])
            s_min = self.polymer_diffusion.kmin*int_samples[i,0,1]
            s_max = self.polymer_diffusion.kmax*int_samples[i,0,1]
            for j in range(int_.shape[1]):
                int_[i,j] = torch.mean(results[:,j])*(s_max-s_min)
        int_ -= 1
        return f_, int_, torch.tensor([0.])