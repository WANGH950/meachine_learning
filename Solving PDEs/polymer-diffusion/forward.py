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
        A[0,0] = -mu
        A[0,1] = mu
        A[nmax,nmax] = -lam
        A[nmax,nmax-1] = lam
        for i in range(nmax-1):
            A[i+1,i] = lam
            A[i+1,i+1] = -(mu+lam)
            A[i+1,i+2] = mu
        self.polymer_diffusion = model.Polymer_Diffusion(A,D0,nmin,nmax, smax,tmin, tmax, alpha, int_Ns, int_Nt, batch_size)
        self.mlp = common.MLP_v1(2,nmax+1,self.polymer_diffusion.kmin,self.polymer_diffusion.kmax)

    def forward(self):
        f_samples, int_samples = self.polymer_diffusion.get_samples()
        f_ = self.polymer_diffusion.f(self.mlp,f_samples[:,0:1],f_samples[:,1:2])
        # 积分限制项
        int_ = torch.ones([self.polymer_diffusion.int_Nt,1])
        # 初值限制项
        u0_ = torch.ones(self.polymer_diffusion.nmax+1)
        for i in range(int_.shape[0]):
            results = self.mlp(int_samples[i])
            s_min = self.polymer_diffusion.kmin*int_samples[i,0,1]
            s_max = self.polymer_diffusion.kmax*int_samples[i,0,1]
            int_[i,0] = torch.sum(results)*(s_max-s_min)/self.polymer_diffusion.int_Ns
            if i == 0:
                for j in range(self.polymer_diffusion.nmax+1):
                    u0_[j] = torch.mean(results[:,j])
                u0_ = u0_*(s_max-s_min) - 1./(self.polymer_diffusion.nmax+1)
        int_ -= 1
        return f_, int_, u0_