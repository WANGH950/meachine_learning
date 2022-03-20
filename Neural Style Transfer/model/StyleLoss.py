import torch
import torch.nn as nn

# 定义风格损失
class StyleLoss(nn.Module):
    def __init__(self,style_features,style_weights):
        super(StyleLoss,self).__init__()
        self.style_features = style_features
        self.style_weights = style_weights
        # 计算风格Gram矩阵
        self.style_grams = self.get_grams(self.style_features)

    # 前向传播
    def forward(self,output_features):
        style_grams = self.get_grams(output_features)
        style_err = torch.tensor(0.).cuda()
        for name, gram in style_grams.items():
            style_err += self.style_weights[name]*torch.sum((gram-self.style_grams[name].detach())**2)
        return style_err
        
    # 计算Gram矩阵
    def get_grams(self,features):
        grams = {}
        for name, feature in features.items():
            _,c,h,w = feature.size()
            feature = feature.view(c,h*w).cuda()
            gram = torch.mm(feature,feature.t())
            grams[name] = gram.div(c*h*w)
        return grams