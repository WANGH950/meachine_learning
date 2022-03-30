import torch
import torch.nn as nn
import blackScholes as model
import sys
sys.path.append("../") 
import common

class deepBSDE_plus(nn.Module):
    def __init__(self,input_dim,R,delta,v_h,v_l,gamma_h,gamma_l,mu,sigma,N,batch_size):
        super(deepBSDE_plus,self).__init__()
        # 存储参数
        self.N = N
        self.batch_size = batch_size
        # 实例化
        self.BS = model.Black_Scholes(input_dim,R,delta,v_h,v_l,gamma_h,gamma_l,mu,sigma)
        # 需要训练的结果
        self.result = nn.Parameter((torch.ones([1])*45).float(),requires_grad=True)
        # 近似梯度
        self.mlp = common.MLP(input_dim+1,input_dim)

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