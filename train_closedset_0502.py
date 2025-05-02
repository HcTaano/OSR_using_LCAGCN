# train_closedset_0502.py

import os
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 从 config.py 导入集中配置（可选）
try:
    from config import train_cfg, path_cfg
    USE_CONFIG = True
except ImportError:
    USE_CONFIG = False

from utils import load_msr, load_utk, split_known_unknown
from models import LCAGCN

def parse_args():
    parser = argparse.ArgumentParser(
        description="Closed-set Training for LC-AGCN with Resume Support")
    parser.add_argument('--dataset', choices=['MSR','UTK'], required=not USE_CONFIG,
                        default=(train_cfg.dataset if USE_CONFIG else None),
                        help="选择数据集：MSR 或 UTK")
    parser.add_argument('--known_classes', type=int, nargs='+',
                        default=(train_cfg.known_classes if USE_CONFIG else None),
                        help="已知类列表，如 0 1 …")
    parser.add_argument('--batch_size', type=int,
                        default=(train_cfg.batch_size if USE_CONFIG else 64))
    parser.add_argument('--epochs', type=int,
                        default=(train_cfg.epochs if USE_CONFIG else 100))
    parser.add_argument('--lr', type=float,
                        default=(train_cfg.lr if USE_CONFIG else 1e-3))
    parser.add_argument('--resume', type=str, default=None,
                        help="Checkpoint 文件路径，继续训练时使用")  # 恢复训练&#8203;:contentReference[oaicite:2]{index=2}
    return parser.parse_args()

def main():
    args = parse_args()

    # 一、加载数据
    if args.dataset == 'MSR':
        data, labels = load_msr(path_cfg.msr_data if USE_CONFIG else "data/MSRAction3DSkeletonReal3D")
    else:
        data, labels = load_utk(
            path_cfg.utk_data if USE_CONFIG else "data/UTKinect_skeletons",
            path_cfg.utk_label if USE_CONFIG else "data/UTKinect_skeletons/actionLabel.txt"
        )

    # 二、划分已知/未知，仅用已知训练
    train_x, train_y, _, _, _, _ = split_known_unknown(data, labels, args.known_classes)
    assert train_x.shape[0] > 0, "训练集为空，请检查 known_classes 是否正确"

    # 三、构造 DataLoader
    # tensor_x = torch.from_numpy(train_x)
    # numpy float32 → torch.FloatTensor；若 train_x 仍为 float64，可强制 .astype 再转
    tensor_x = torch.from_numpy(train_x.astype(np.float32)).float()  # 强制 float32
    # numpy→tensor
    tensor_y = torch.from_numpy(train_y).long()
    loader = DataLoader(TensorDataset(tensor_x, tensor_y),
                        batch_size=args.batch_size, shuffle=True)

    # 四、模型、损失、优化器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LCAGCN(num_class=len(args.known_classes)).to(device)  # 类别数由已知类长度决定
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 五、可选：从 checkpoint 恢复
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optim_state'])           # 恢复 Adam 内部状态&#8203;:contentReference[oaicite:3]{index=3}
        start_epoch = ckpt['epoch'] + 1
        print(f"Resumed training from epoch {start_epoch}")

    # 六、训练循环
    model.train()
    for epoch in range(start_epoch, args.epochs):
        total_loss = 0.0
        for x, y in loader:
            # x, y = x.to(device), y.to(device)
            # 再次确保输入为 float32，再送入模型&#8203;:contentReference[oaicite:0]{index=0}
            x = x.to(device).float()
            y = y.to(device)
            logits, _ = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        print(f"[{args.dataset}] Epoch {epoch}/{args.epochs-1}  Loss: {avg_loss:.4f}")

        # 七、保存 checkpoint
        os.makedirs(path_cfg.ckpt_dir if USE_CONFIG else "checkpoints", exist_ok=True)
        ckpt_path = os.path.join(
            path_cfg.ckpt_dir if USE_CONFIG else "checkpoints",
            f"{args.dataset.lower()}_checkpoint_epoch{epoch}.pth"
        )
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optim_state': optimizer.state_dict()
        }, ckpt_path)

    # 八、最终保存模型权重
    final_path = os.path.join(
        path_cfg.ckpt_dir if USE_CONFIG else "checkpoints",
        f"{args.dataset.lower()}_model_known{len(args.known_classes)}.pth"
    )
    torch.save(model.state_dict(), final_path)
    print("Training complete. Model saved to", final_path)

if __name__ == "__main__":
    main()
