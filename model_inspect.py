# model_inspect.py

import torch
from torchsummary import summary    # 需要先 pip install torchsummary
from models.lc_agcn_old import LCAGCN

def count_parameters(model):
    # 返回总的可训练参数量
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # 假设使用 18 个已知类（MSR 数据集）
    num_known = 18
    model = LCAGCN(num_class=num_known)

    # 将模型移动到 CPU 并打印整体结构与每层输出尺寸、参数量
    device = torch.device('cpu')
    model.to(device)
    print("=== LCAGCN 模型结构 ===")
    summary(model, input_size=(3,20,20), device='cpu')
    # input_size=(C,T,V) 即 3 通道, 20 帧, 20 关节

    # 打印总的可训练参数量
    total_params = count_parameters(model)
    print(f"\nTotal trainable parameters: {total_params:,}")
