import torch
import torch.nn as nn
import torch.nn.functional as F

class LCAGCN(nn.Module):
    """
    增强版的 LC-AGCN: 增加通道容量和残差连接
    """
    def __init__(self, num_class=10, num_point=20):
        super().__init__()
        # 增加通道数
        self.conv_spatial = nn.Conv2d(3, 128, kernel_size=1)
        # 多尺度时域卷积，增加不同kernel尺寸捕获时序特征
        self.conv_t1 = nn.Conv2d(128, 64, kernel_size=(3,1), padding=(1,0))
        self.conv_t2 = nn.Conv2d(128, 64, kernel_size=(9,1), padding=(4,0))
        
        # 额外卷积层增加表示能力
        self.conv_extra = nn.Conv2d(128, 128, kernel_size=(3,1), padding=(1,0))
        
        # Batch Normalization提高训练稳定性
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(128)
        
        # 注意力机制捕获关键关节
        self.joint_attention = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 分类器层
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(0.5)  # 防止过拟合
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_class)

    def forward(self, x):
        # x: [N, C, T, V]
        x = self.conv_spatial(x)        # [N,128,T,V]
        x = F.relu(self.bn1(x))
        
        # 多尺度时域卷积
        x1 = self.conv_t1(x)
        x2 = self.conv_t2(x)
        x = torch.cat([x1, x2], dim=1)  # [N,128,T,V]
        
        # 残差连接
        identity = x
        x = self.conv_extra(x)
        x = F.relu(self.bn2(x)) + identity
        
        # 应用关节注意力
        att = self.joint_attention(x)
        x = x * att
        
        # 特征提取
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        features = x                    # 保存特征向量
        
        # 分类器
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        out = self.fc2(x)
        
        return out, features