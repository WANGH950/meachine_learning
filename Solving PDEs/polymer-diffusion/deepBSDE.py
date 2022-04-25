import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append('../')
import common

class PolymerDiffusion:
    def __init__(self, lam, mu, D0, nmin, nmax, alpha, batch_size):
        self.lam = lam
        self.mu = mu
        self.D0 = D0
        self.nmin = nmin
        self.nmax = nmax
        self.alpha = alpha
        self.batch_size = batch_size

    def u0(self,n,s):
        beta = 1000000
        return torch.sqrt(torch.tensor(beta/2/np.pi))*torch.exp(-beta*s**2/2)/(self.nmax+1)

    def U(self,x):
        return self.D0 / (x+self.nmin)**self.alpha
    
    def delta_N(self,pre_n,delta_t,forward=True,k = 20):
        Un = torch.rand(k)
        log_res = -torch.log(Un)
        lambda_k = log_res/self.lam/2
        mu_k = log_res/self.mu/2
        lambda_sk = torch.zeros(k+1)
        mu_sk = torch.zeros(k+1)
        for i in range(k):
            lambda_sk[i+1] = lambda_sk[i] + lambda_k[i]
            mu_sk[i+1] = mu_sk[i] + mu_k[i]
        delta_n = torch.zeros_like(pre_n).float()
        rand_eta = torch.rand(pre_n.shape[0])
        for i in range(pre_n.shape[0]):
            if rand_eta[i] < 0.5:
                for j in range(k):
                    if delta_t < lambda_sk[j+1]:
                        if forward and pre_n[i,0] <= self.nmax-j:
                            delta_n[i,0] = j
                        elif not forward and pre_n[i,0] >= j:
                            delta_n[i,0] = j
                        break
            else:
                for j in range(k):
                    if delta_t < mu_sk[j+1]:
                        if forward and pre_n[i,0] >= j:
                            delta_n[i,0] = -j
                        elif not forward and pre_n[i,0] <= self.nmax-j:
                            delta_n[i,0] = -j
                        break
        return delta_n

    # 正反向采样（确保充分利用初值条件）
    def get_samples(self,delta_t,N):
        # 正向采样
        samples_X1 = torch.randint(0,self.nmax+1,[self.batch_size,N+1]).float()
        samples_X2 = torch.zeros_like(samples_X1)
        samples_deltaB = torch.randn([self.batch_size,N])*torch.sqrt(torch.tensor(delta_t))
        for i in range(N):
            samples_X1[:,i+1:i+2] = samples_X1[:,i:i+1] + self.delta_N(samples_X1[:,i:i+1],delta_t,forward=True,k=5)
            samples_X2[:,i+1:i+2] = samples_X2[:,i:i+1] + self.U(samples_X1[:,i:i+1])*delta_t + samples_deltaB[:,i:i+1]
        # 反向采样
        samples_X1_ = torch.randint_like(samples_X1,self.nmax+1).float()
        samples_X2_ = torch.zeros_like(samples_X2)
        samples_deltaB_ = torch.randn([self.batch_size,N])*torch.sqrt(torch.tensor(delta_t))
        samples_X1_[:,N] = samples_X1[:,N]
        samples_X2_[:,N] = samples_X2[:,N]
        for i in range(N):
            samples_X1_[:,N-i-1:N-i] = samples_X1_[:,N-i:N-i+1] - self.delta_N(samples_X1_[:,N-i:N-i+1],delta_t,forward=False,k=5)
            samples_X2_[:,N-i-1:N-i] = samples_X2_[:,N-i:N-i+1] - self.U(samples_X1_[:,N-i:N-i+1])*delta_t - samples_deltaB_[:,N-i-1:N-i]
        return torch.cat([samples_X1,samples_X1_],dim=0), torch.cat([samples_X2,samples_X2_],dim=0), torch.cat([samples_deltaB,samples_deltaB_],dim=0)

    @staticmethod
    def train(model,epoch,t,N,s,n):
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.MSELoss()

        lossu_values = np.zeros(epoch)
        result_values = np.zeros(epoch)

        for i in range(epoch):
            model.train()
            optimizer.zero_grad()
            u,u0 = model(t,N)
            loss_u = criterion(u,u0)
            loss = loss_u
            loss.backward()
            optimizer.step()
            lossu_values[i] = loss_u.item()
            model.eval()
            result_values[i] = model.mlp(torch.tensor([[s,t]]).float())[0,n]
            print('epoch:',i,' lossu:',lossu_values[i],' result:',result_values[i])
        return lossu_values, result_values


class deepBSDEplus(nn.Module):
    def __init__(self, lam, mu, D0, nmin, nmax, alpha, batch_size):
        super(deepBSDEplus,self).__init__()
        self.func = PolymerDiffusion(lam,mu,D0,nmin,nmax,alpha,batch_size)
        self.kmin = self.func.U(nmax)
        self.kmax = self.func.U(0)
        # 近似结果
        self.mlp = common.MLP(2,nmax+1)

    def forward(self,t,N):
        delta_t = t/N
        samples_n, samples_s, samples_deltaB = self.func.get_samples(delta_t,N)
        tk = torch.ones([samples_s.shape[0],1])*t
        u, partial_s, partial_ss = self.get_partial(samples_n[:,N:N+1],samples_s[:,N:N+1],tk)
        for i in range(N):
            u = u - partial_s*samples_deltaB[:,N-i-1:N-i] - partial_ss*delta_t/2
            _, partial_s, partial_ss = self.get_partial(samples_n[:,N-i-1:N-i],samples_s[:,N-i-1:N-i],tk)
            tk -= delta_t
        return u, self.func.u0(samples_n[:,0:1],samples_s[:,0:1])

    def get_partial(self,n,s,t):
        s.requires_grad_(True)
        outputs = self.mlp(torch.cat([s,t],dim=1))
        grad_outputs = torch.zeros_like(outputs)
        for i in range(n.shape[0]):
            grad_outputs[i,n[i,0].int()] = 1
        grad_outputs = grad_outputs.unsqueeze(-1)
        outputs = outputs.unsqueeze(1)
        outputs = torch.bmm(outputs,grad_outputs).squeeze(-1)
        partial_s = torch.autograd.grad(
            outputs=outputs,
            inputs=s,
            grad_outputs=torch.ones_like(outputs),
            retain_graph=True,
            create_graph=True
        )[0]
        partial_ss = torch.autograd.grad(
            outputs=partial_s,
            inputs=s,
            grad_outputs=torch.ones_like(partial_s),
            retain_graph=True,
            create_graph=True
        )[0]
        return outputs, partial_s, partial_ss