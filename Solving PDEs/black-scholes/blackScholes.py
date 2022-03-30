import torch
import numpy as np
import torch.nn as nn

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