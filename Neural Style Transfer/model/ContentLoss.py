import torch
import torch.nn as nn

# 定义内容损失
class ContentLoss(nn.Module):
    def __init__(self,content_feature):
        super(ContentLoss,self).__init__()
        self.content_feature = content_feature
        self.criterion = nn.MSELoss()
    
    # 前向传播
    def forward(self,output_feature):
        content_err = torch.tensor(0.).cuda()
        for name, feature in output_feature.items():
            content_err += self.criterion(feature,self.content_feature[name].detach())
        return content_err
        