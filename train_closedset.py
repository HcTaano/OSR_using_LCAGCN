# train_closedset.py

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from utils import load_msr, load_utk, split_known_unknown
from models import LCAGCN

def main():
    parser = argparse.ArgumentParser(description="闭集训练：LC-AGCN on MSR/UTK")  # argparse 用法
    parser.add_argument('--dataset', choices=['MSR','UTK'], required=True)
    parser.add_argument('--known_classes', type=int, nargs='+', required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    # 根据选择加载数据
    if args.dataset == 'MSR':
        data, labels = load_msr("data/MSRAction3DSkeletonReal3D")
    else:
        data, labels = load_utk(
            "data/UTKinect_skeletons/joints",  # ← 指向真正存放骨架 .txt 的子文件夹
            "data/UTKinect_skeletons/actionLabel.txt"
        )

    # 划分仅已知类进行训练
    train_x, train_y, *_ = split_known_unknown(data, labels, args.known_classes)

    # DataLoader 准备
    # ds = TensorDataset(torch.tensor(train_x), torch.tensor(train_y))
    # 对数据直接从 numpy→tensor，保持 dtype 和内存共享
    ds = TensorDataset(torch.from_numpy(train_x),  # numpy.float32 → torch.float32 :contentReference[oaicite:2]{index=2}
                       torch.from_numpy(train_y).long())  # numpy.int64 → torch.int64

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    # 模型、损失与优化器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LCAGCN(num_class=len(args.known_classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 训练循环
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {total_loss/len(loader):.4f}")

    # 保存模型权重
    torch.save(model.state_dict(),
               f"checkpoints/{args.dataset.lower()}_model_known{len(args.known_classes)}.pth")
    print("模型已保存。")

if __name__ == "__main__":
    main()



# # File: train_closedset.py
# import torch, torch.nn as nn, torch.optim as optim
# from torch.utils.data import TensorDataset, DataLoader
# from models.lc_agcn import LCAGCN
# from utils.data_utils import load_msr, load_utk
#
# def train_model(data, labels, num_classes, device):
#     model = LCAGCN(num_class=num_classes).to(device)
#     optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
#     criterion = nn.CrossEntropyLoss()
#     dataset = TensorDataset(torch.tensor(data), torch.tensor(labels))
#     loader = DataLoader(dataset, batch_size=16, shuffle=True)
#     model.train()
#     for epoch in range(30):
#         for x, y in loader:
#             x, y = x.to(device), y.to(device)
#             logits, _ = model(x)
#             loss = criterion(logits, y)
#             optimizer.zero_grad(); loss.backward(); optimizer.step()
#     return model
#
# if __name__ == "__main__":
#     import sys
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     # 训练 MSRAction3D (20 类闭集)
#     msr_data, msr_labels = load_msr("data/MSRAction3DSkeletonReal3D")
#     msr_model = train_model(msr_data, msr_labels, num_classes=20, device=device)
#     torch.save(msr_model.state_dict(), "models/msr_model.pth")
#     # 训练 UTKinect (10 类闭集)
#     utk_data, utk_labels = load_utk("data/UTKinect_skeletons", "data/actionLabel.txt")
#     utk_model = train_model(utk_data, utk_labels, num_classes=10, device=device)
#     torch.save(utk_model.state_dict(), "models/utk_model.pth")
