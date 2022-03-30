import torch
import numpy as np
import torch.autograd.functional as functional

class Polymer_Diffusion:
    def __init__(self, A, D0, nmin, nmax, smax,tmin, tmax, alpha, int_Ns, int_Nt, batch_size):
        # 储存参数
        self.A = A
        self.D0 = D0
        self.nmin = nmin
        self.nmax = nmax
        self.smax = smax
        self.tmin = tmin
        self.tmax = tmax
        self.alpha = alpha
        self.int_Ns = int_Ns
        self.int_Nt = int_Nt
        self.batch_size = batch_size
        self.kmin = self.U(nmax)
        self.kmax = self.U(0)
        self.Un = self.U(torch.tensor([i for i in range(nmax+1)]))

    def U(self,n):
        return self.D0 / (n + self.nmin)**self.alpha

    # 计算f
    def f(self,mlp,s,t):
        s.requires_grad_(True)
        t.requires_grad_(True)
        outputs = mlp(torch.cat([s,t],dim=1))
        mlp_outputs = mlp.mlp_outputs
        (a,b) = functional.jacobian(
            func=lambda x,y:mlp(torch.cat([x,y],dim=1)),
            inputs=(s,t),
            create_graph=True
        )
        u_s = torch.diagonal(a[:,:,:,0],dim1=0,dim2=2).T
        u_t = torch.diagonal(b[:,:,:,0],dim1=0,dim2=2).T
        limits_s = torch.zeros([outputs.shape[0],1])
        limits_t = torch.zeros([outputs.shape[0],1])
        for i in range(outputs.shape[0]):
            if t[i,0] > 0 and s[i,0] > self.kmin*t[i,0] and s[i,0] < self.kmax*t[i,0]:
                limits_s[i,0] = ((self.kmax+self.kmin)*t[i,0] - 2*s[i,0])/(self.kmin*t[i,0])**2
                limits_t[i,0] = (2*s[i,0]/t[i,0] - self.kmin - self.kmax)*s[i,0]/(self.kmin*t[i,0])**2
        return u_t - torch.mm(outputs,self.A.T) + u_s*self.Un + (limits_t + self.Un*limits_s)*mlp_outputs

    # 采样函数
    def get_samples(self):
        f_t = torch.rand([self.batch_size,1])*(self.tmax-self.tmin)+self.tmin
        f_s = torch.rand([self.batch_size,1])
        for i in range(self.batch_size):
            s_min = self.kmin*f_t[i]
            s_max = self.kmax*f_t[i]
            f_s[i,0] = f_s[i,0]*(s_max-s_min) + s_min
        f_samples = torch.cat([f_s,f_t],dim=1)

        # 积分条件分段采样, 仅采样定义域内的值
        int_sample_t = torch.linspace(self.tmin,self.tmax,self.int_Nt)
        int_samples = torch.ones([self.int_Nt,self.int_Ns,2])
        for i in range(self.int_Nt):
            int_samples[i] = torch.cat([torch.linspace(torch.min(torch.tensor([self.kmin*int_sample_t[i],self.smax])),torch.min(torch.tensor([self.kmax*int_sample_t[i],self.smax])),self.int_Ns).unsqueeze(-1),torch.ones([self.int_Ns,1])*int_sample_t[i]],dim=1)
        return f_samples, int_samples

    # 训练器
    @staticmethod
    def train_PINNs_v1(model,epoch,w_f,w_int,w_u0):
        optimizer = torch.optim.Adam(model.parameters())
        loss_f_values = np.zeros(epoch)
        loss_int_values = np.zeros(epoch)
        loss_u0_values = np.zeros(epoch)
        for i in range(epoch):
            model.train()
            optimizer.zero_grad()
            f_,int_,u0_ = model()
            loss_f = torch.mean(f_**2)
            loss_int = torch.mean(int_**2)
            loss_u0 = torch.sum(u0_**2)
            loss = loss_f*w_f+loss_int*w_int+loss_u0*w_u0
            loss.backward()
            optimizer.step()
            loss_f_values[i] = loss_f.item()
            loss_int_values[i] = loss_int.item()
            loss_u0_values[i] = loss_u0.item()
            print('epoch:',i)
            print('loss_f:',loss_f_values[i],' loss_int:',loss_int_values[i],' loss_u0:',loss_u0_values[i])

        return loss_f_values,loss_int_values,loss_u0_values