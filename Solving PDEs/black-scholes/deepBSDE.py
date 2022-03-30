import torch
import torch.nn as nn
import blackScholes as model
import sys
sys.path.append("../") 
import common

class deepBSDE(nn.Module):
    def __init__(self,input_dim,R,delta,v_h,v_l,gamma_h,gamma_l,mu,sigma,N,batch_size):
        super(deepBSDE,self).__init__()
        # 存储参数
        self.N = N
        self.batch_size = batch_size
        # 实例化
        self.BS = model.Black_Scholes(input_dim,R,delta,v_h,v_l,gamma_h,gamma_l,mu,sigma)
        # 需要训练的结果
        self.result = nn.Parameter((torch.ones([1])*45).float(),requires_grad=True)
        # 梯度
        self.result_grad = nn.Parameter(torch.rand([1,input_dim]).float(),requires_grad=True)
        # 近似后n-1个梯度
        self.mlp = nn.ModuleList([common.MLP(input_dim,input_dim) for _ in range(N-1)])

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