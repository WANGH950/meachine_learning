import torch
import numpy as np
import torch.nn as nn
import torch.autograd.functional as functional
import sys
sys.path.append("../") 
import common

# 定义Black_Scholes的BSDE迭代器
class Black_Scholes:
    def __init__(self,input_dim,R,delta,v_h,v_l,gamma_h,gamma_l,mu,sigma):
        # 存储参数
        self.input_dim = input_dim
        self.R = R
        self.delta = delta
        self.v_h = v_h
        self.v_l = v_l
        self.gamma_h = gamma_h
        self.gamma_l = gamma_l
        self.mu = mu
        self.sigma = sigma
        self.slope = (self.gamma_h - self.gamma_l) / (self.v_h - self.v_l)
    
    # 采样函数
    def get_samples(self,t,x,T,N):
        batch_size = x.shape[0]
        delta_t = (T-t)/N
        delta_B_sampels = torch.randn([batch_size,N,self.input_dim])*torch.sqrt(delta_t)
        X_samples = torch.zeros([batch_size,N+1,self.input_dim])
        X_samples[:,0] += x

        for i in range(N):
            X_samples[:,i+1] = X_samples[:,i] + self.mu*X_samples[:,i]*delta_t + self.sigma*X_samples[:,i]*delta_B_sampels[:,i]

        return delta_t, delta_B_sampels, X_samples

    # 强度函数
    def Q_(self, y):
        batch_size = y.shape[0]
        result = torch.zeros([batch_size,1])
        for i in range(batch_size):
            if y[i,0] < self.v_h:
                result[i,0] += self.gamma_h
            elif y[i,0] >= self.v_l:
                result[i,0] += self.gamma_l
            else:
                result[i,0] += self.slope * (y[i,0]-self.v_h)+self.gamma_h
        return ((1-self.delta)*result+ self.R) * y
    
    # 边值条件
    def g_(self, x):
        batch_size = x.shape[0]
        result = torch.ones([batch_size,1])
        for i in range(batch_size):
            result[i,0] = torch.min(x[i,:])
        return result

    # 训练器
    @staticmethod
    def train_BSDEs(model,epoch,t,x,T):
        # 优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=0.008)
        # 损失函数
        criterion = nn.MSELoss()
        # 用来保存训练过程产生的数据
        loss_values = np.zeros(epoch)
        result_values = np.zeros(epoch)
        relative_errs = np.zeros(epoch)

        final_result = .0
        final_relative_err = .0
        for i in range(epoch):
            model.train()
            train_loss = .0
            optimizer.zero_grad()
            gX_n,u = model(t,x,T)
            final_result = model.result
            loss = criterion(u,gX_n)
            final_relative_err = torch.abs(final_result[0]-57.300)/57.300
            loss.backward()
            optimizer.step()
            train_loss = loss.item()

            # 保存训练过程的数据
            loss_values[i] = train_loss
            result_values[i] = final_result[0]
            relative_errs[i] = final_relative_err
            print('epoch:',i,' loss:',train_loss,' result:',final_result[0].detach().numpy(),' relative:',final_relative_err.detach().numpy())
        return loss_values, result_values, relative_errs, final_result, final_relative_err

    # 训练器
    @staticmethod
    def train_PINNs(model,epoch,t,x,T):
        # 优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=0.008)
        # 损失函数
        criterion = nn.MSELoss()
        # 用来保存训练过程产生的数据
        loss_values = np.zeros(epoch)
        result_values = np.zeros(epoch)

        final_result = .0
        for i in range(epoch):
            model.train()
            train_loss = .0
            optimizer.zero_grad()
            f,gX_n,u = model(t,x,T)
            final_result = model.mlp(torch.cat([x,t],dim=1))[0,0]
            loss = criterion(u,gX_n) + torch.mean(f**2)
            loss.backward()
            optimizer.step()
            train_loss = loss.item()

            # 保存训练过程的数据
            loss_values[i] = train_loss
            result_values[i] = final_result
            print('epoch:',i,' loss:',train_loss,' result:',final_result.cpu().detach().numpy())
        return loss_values, result_values, final_result


# 原始算法
class deepBSDE(nn.Module):
    def __init__(self,input_dim,R,delta,v_h,v_l,gamma_h,gamma_l,mu,sigma,N,batch_size):
        super(deepBSDE,self).__init__()
        # 存储参数
        self.N = N
        self.batch_size = batch_size
        # 实例化
        self.BS = Black_Scholes(input_dim,R,delta,v_h,v_l,gamma_h,gamma_l,mu,sigma)
        # 需要训练的结果
        self.result = nn.Parameter((torch.ones([1])*45).float(),requires_grad=True)
        # 梯度
        self.result_grad = nn.Parameter(torch.rand([1,input_dim]).float(),requires_grad=True)
        # 近似后n-1个梯度
        self.mlp = nn.ModuleList([common.MLP(input_dim,input_dim) for _ in range(N-1)])

    # t: 目标时间 1*1
    # x: 目标位置 1*d
    # T: 终端时间 1*1
    def forward(self,t,x,T):
        ones = torch.ones([self.batch_size,1]).float()
        u = ones * self.result
        gradient_u = torch.matmul(ones,self.result_grad)
        # 获取样本数据
        delta_t,delta_B,X_n = self.BS.get_samples(t,torch.matmul(ones,x),T,self.N)

        for i in range(self.N-1):
            u = u + self.BS.Q_(u)*delta_t + (gradient_u*delta_B[:,i]).sum(1).reshape([self.batch_size,1])
            gradient_u = self.mlp[i](X_n[:,i+1])

        u = u + self.BS.Q_(u)*delta_t +(gradient_u*delta_B[:,self.N-1]).sum(1).reshape([self.batch_size,1])
        
        return self.BS.g_(X_n[:,self.N]), u


# 改进算法
class deepBSDE_plus(nn.Module):
    def __init__(self,input_dim,R,delta,v_h,v_l,gamma_h,gamma_l,mu,sigma,N,batch_size):
        super(deepBSDE_plus,self).__init__()
        # 存储参数
        self.N = N
        self.batch_size = batch_size
        # 实例化
        self.BS = Black_Scholes(input_dim,R,delta,v_h,v_l,gamma_h,gamma_l,mu,sigma)
        # 需要训练的结果
        self.result = nn.Parameter((torch.ones([1])*45).float(),requires_grad=True)
        # 近似梯度
        self.mlp = common.MLP(input_dim+1,input_dim)

    # t: 目标时间 1*1
    # x: 目标位置 1*d
    # T: 终端时间 1*1
    def forward(self,t,x,T):
        ones = torch.ones([self.batch_size,1]).float()
        # 获取样本数据
        delta_t,delta_B,X_n = self.BS.get_samples(t,torch.matmul(ones,x),T,self.N)
        u = ones * self.result
        t = ones * t

        for i in range(self.N):
            gradient_u = self.mlp(torch.cat((X_n[:,i],t),dim=1))
            u = u + self.BS.Q_(u)*delta_t + (gradient_u*delta_B[:,i]).sum(1).reshape([self.batch_size,1])
        
        return self.BS.g_(X_n[:,self.N]), u


class PINN_v1(nn.Module):
    def __init__(self,input_dim,output_dim,R,delta,v_h,v_l,gamma_h,gamma_l,mu,sigma,N,batch_size):
        super(PINN_v1,self).__init__()
        self.N = N
        self.batch_size = batch_size
        # 实例化函数对象
        self.BS = Black_Scholes(input_dim,R,delta,v_h,v_l,gamma_h,gamma_l,mu,sigma)
        # MLP近似u
        self.mlp = common.MLP(input_dim+1,output_dim)
        # 优化处理，增加leakyReLU层
        self.mlp.add_module('leakyReLU',nn.LeakyReLU(0.05))

    def forward(self,t,x,T):
        # 获取样本数据
        X = torch.zeros([self.batch_size,self.N+1,self.BS.input_dim])
        delta_t = (T-t)/self.N
        t = torch.ones([self.batch_size,1])*t
        f = torch.zeros([self.batch_size,self.N])
        # 从目标点附近正态分布范围取batch条数据，按照BSDE采样方式采样X（维度高，均匀取的话不现实）
        for i in range(self.batch_size):
            _,_,X[i:i+1] = self.BS.get_samples(t[0,0],x+torch.randn([1,self.BS.input_dim]),T,self.N)
        
        for i in range(self.N):
            f[:,i:i+1] = self.f(X[:,i],t)
            t = t + delta_t
        outputs = self.mlp(torch.cat([X[:,self.N],t],dim=1))
        return f, self.BS.g_(X[:,self.N]),outputs

    def f(self,x,t):
        t.requires_grad_(True)
        x.requires_grad_(True)
        outputs = self.mlp(torch.cat([x,t],dim=1))
        u_t = torch.autograd.grad(
            outputs=outputs,
            inputs=t,
            grad_outputs=torch.ones_like(outputs),
            retain_graph=True,
            create_graph=True
        )[0]
        u_x = torch.autograd.grad(
            outputs=outputs,
            inputs=x,
            grad_outputs=torch.ones_like(outputs),
            retain_graph=True,
            create_graph=True
        )[0]
        u_hessian = torch.zeros([self.batch_size,self.BS.input_dim,self.BS.input_dim])
        for i in range(self.batch_size):
            u_hessian[i] += functional.hessian(lambda x:self.mlp(torch.cat([x,t[i:i+1]],dim=1))[0,0],x[i:i+1],True).reshape([self.BS.input_dim,self.BS.input_dim])
        res = torch.zeros([self.batch_size,1])
        for i in range(self.batch_size):
            res[i,0] = u_t[i,0] + self.BS.mu*torch.dot(x[i],u_x[i]) + self.BS.sigma**2/2*torch.trace(u_hessian[i]*x[i:i+1].T**2) - self.BS.Q_(outputs[i:i+1])
        return res