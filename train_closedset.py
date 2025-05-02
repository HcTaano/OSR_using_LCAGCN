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

# 添加数据增强类
class RotationTransform:
    def __init__(self, max_degree=10):
        self.max_degree = max_degree
        
    def __call__(self, x):
        # x: [B, C, T, V]
        batch_size = x.shape[0]
        angles = torch.rand(batch_size) * 2 * self.max_degree - self.max_degree
        angles = angles * (np.pi / 180)  # 转为弧度
        
        # 绕z轴旋转矩阵
        rot_mats = []
        for angle in angles:
            cos_a, sin_a = torch.cos(angle), torch.sin(angle)
            rot_mat = torch.tensor([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1]
            ], device=x.device)
            rot_mats.append(rot_mat)
        
        rot_mats = torch.stack(rot_mats, dim=0).to(x.device)
        
        # 应用旋转到每一帧
        x_new = torch.zeros_like(x)
        for b in range(batch_size):
            for t in range(x.shape[2]):  # 遍历每一帧
                for v in range(x.shape[3]):  # 遍历每个关节点
                    # 对每个关节点的 3D 坐标应用旋转
                    # 提取单个关节的 [3] 向量，旋转后放回
                    coords = x[b, :, t, v]  # [3]
                    rotated_coords = torch.matmul(rot_mats[b], coords)  # [3,3] x [3] -> [3]
                    x_new[b, :, t, v] = rotated_coords
        
        return x_new

class NoiseTransform:
    def __init__(self, scale=0.02):
        self.scale = scale
        
    def __call__(self, x):
        # 添加高斯噪声
        noise = torch.randn_like(x) * self.scale
        return x + noise

class ScaleTransform:
    def __init__(self, range=(0.9, 1.1)):
        self.min_scale, self.max_scale = range
        
    def __call__(self, x):
        # 随机缩放
        batch_size = x.shape[0]
        scales = torch.rand(batch_size, 1, 1, 1, device=x.device) * (self.max_scale - self.min_scale) + self.min_scale
        return x * scales

class TemporalCrop:
    def __init__(self, min_ratio=0.8):
        self.min_ratio = min_ratio
        
    def __call__(self, x):
        # 时序裁剪: 随机保留部分时间段，其余补零
        batch_size, _, t_len, _ = x.shape
        new_x = x.clone()
        
        for i in range(batch_size):
            keep_len = int(t_len * (torch.rand(1).item() * (1 - self.min_ratio) + self.min_ratio))
            start = torch.randint(0, t_len - keep_len + 1, (1,)).item()
            # 将非保留部分置零
            mask = torch.zeros(t_len, device=x.device)
            mask[start:start+keep_len] = 1
            mask = mask.view(1, 1, -1, 1)
            new_x[i] = new_x[i] * mask
            
        return new_x

def apply_transforms(x, transforms):
    """应用一系列数据增强"""
    for transform in transforms:
        x = transform(x)
    return x

def compute_class_weights(labels):
    """根据类别频率计算权重，频率越低权重越高"""
    class_counts = np.bincount(labels)
    n_samples = len(labels)
    n_classes = len(class_counts)
    weights = n_samples / (n_classes * class_counts)
    # 确保返回float32类型的权重，避免类型不匹配
    return weights.astype(np.float32)

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
                        help="Checkpoint 文件路径，继续训练时使用")
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
    train_x, train_y, val_x, val_y, _, _ = split_known_unknown(data, labels, args.known_classes)
    assert train_x.shape[0] > 0, "训练集为空，请检查 known_classes 是否正确"

    # 三、构造 DataLoader
    # tensor_x = torch.from_numpy(train_x)
    # numpy float32 → torch.FloatTensor；若 train_x 仍为 float64，可强制 .astype 再转
    tensor_x = torch.from_numpy(train_x.astype(np.float32)).float()  # 强制 float32
    # numpy→tensor
    tensor_y = torch.from_numpy(train_y).long()
    train_loader = DataLoader(TensorDataset(tensor_x, tensor_y),
                        batch_size=args.batch_size, shuffle=True)
    
    # 验证集
    if val_x.shape[0] > 0:
        val_tensor_x = torch.from_numpy(val_x.astype(np.float32)).float()
        val_tensor_y = torch.from_numpy(val_y).long()
        val_loader = DataLoader(TensorDataset(val_tensor_x, val_tensor_y),
                            batch_size=args.batch_size, shuffle=False)
    else:
        val_loader = None


    # 四、模型、损失、优化器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LCAGCN(num_class=len(args.known_classes)).to(device)  # 类别数由已知类长度决定
    
    # 计算类别权重
    class_weights = compute_class_weights(train_y)
    # 确保权重是float32类型，与模型参数类型保持一致
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    # 使用AdamW和余弦退火学习率调度器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2
    )

    # 在train_closedset.py中添加以下代码以检查数据
    print(f"数据形状: {train_x.shape}, 类型: {train_x.dtype}")
    print(f"标签形状: {train_y.shape}, 类型: {train_y.dtype}")
    print(f"标签分布: {np.bincount(train_y)}")

    # 五、可选：从 checkpoint 恢复
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optim_state'])
        start_epoch = ckpt['epoch'] + 1
        print(f"Resumed training from epoch {start_epoch}")

    # 创建数据增强
    transforms = [
        RotationTransform(max_degree=10),  # 随机旋转
        NoiseTransform(scale=0.02),        # 随机噪声
        ScaleTransform(range=(0.9, 1.1)),  # 随机缩放
        TemporalCrop(min_ratio=0.8),       # 时序裁剪
    ]
    
    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler()
    
    # 早停策略
    patience = 50
    best_val_acc = 0
    no_improve_epochs = 0
    
    # 六、训练循环
    model.train()
    for epoch in range(start_epoch, args.epochs):
        total_loss = 0.0
        for x, y in train_loader:
            # 数据增强
            x = x.to(device).float()
            y = y.to(device)
            x = apply_transforms(x, transforms)
            
            optimizer.zero_grad()
            # 使用混合精度训练
            with torch.cuda.amp.autocast():
                logits, _ = model(x)
                loss = criterion(logits, y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
        # 学习率调整
        scheduler.step()
        
        avg_loss = total_loss / len(train_loader)
        print(f"[{args.dataset}] Epoch {epoch}/{args.epochs-1}  Loss: {avg_loss:.4f}")

        # 验证阶段
        if val_loader is not None:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device).float()
                    y = y.to(device)
                    logits, _ = model(x)
                    _, pred = torch.max(logits, 1)
                    total += y.size(0)
                    correct += (pred == y).sum().item()
            
            val_acc = correct / total
            print(f"Validation Accuracy: {val_acc:.4f}")
            
            # 早停检查
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                # 七、保存最佳模型
                os.makedirs(path_cfg.ckpt_dir if USE_CONFIG else "checkpoints", exist_ok=True)
                best_model_path = os.path.join(
                    path_cfg.ckpt_dir if USE_CONFIG else "checkpoints",
                    f"{args.dataset.lower()}_best_model.pth"
                )
                torch.save({
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optim_state': optimizer.state_dict(),
                    'val_acc': val_acc
                }, best_model_path)
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= patience:
                    print(f"早停: {patience}轮无改善")
                    break
            
            model.train()
        
        # 每轮保存 checkpoint
        if epoch % 5 == 0 or epoch == args.epochs - 1:
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

