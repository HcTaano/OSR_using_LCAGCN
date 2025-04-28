# File: models/lc_agcn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class LCAGCN(nn.Module):
    """
    简化版的 LC-AGCN: 通道卷积 + 时域多尺度卷积。
    输入格式: x [N, C, T, V] (C=3坐标通道, V=关节数)
    """
    def __init__(self, num_class=10, num_point=20):
        super().__init__()
        # 通道/空间卷积 (1x1)，扩张输入通道到高维
        self.conv_spatial = nn.Conv2d(3, 64, kernel_size=1)
        # 多尺度时域卷积 (9帧核，padding=4 保持长度)
        self.conv_temporal = nn.Conv2d(64, 64, kernel_size=(9,1), padding=(4,0))
        # 池化和全连接分类
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64, num_class)

    def forward(self, x):
        # x: [N, C, T, V]
        x = self.conv_spatial(x)        # [N,64,T,V]
        x = F.relu(x)
        x = self.conv_temporal(x)       # [N,64,T,V]
        x = F.relu(x)
        # 池化至 [N,64,1,1]
        x = self.pool(x)
        x = x.view(x.size(0), -1)       # [N,64]
        features = x                    # penultimate特征
        out = self.fc(x)               # [N,num_class] logits
        return out, features
