# File: train_closedset.py
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from models.lc_agcn import LCAGCN
from utils.data_utils import load_msr, load_utk

def train_model(data, labels, num_classes, device):
    model = LCAGCN(num_class=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    dataset = TensorDataset(torch.tensor(data), torch.tensor(labels))
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    model.train()
    for epoch in range(30):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    return model

if __name__ == "__main__":
    import sys
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 训练 MSRAction3D (20 类闭集)
    msr_data, msr_labels = load_msr("data/MSRAction3DSkeletonReal3D")
    msr_model = train_model(msr_data, msr_labels, num_classes=20, device=device)
    torch.save(msr_model.state_dict(), "models/msr_model.pth")
    # 训练 UTKinect (10 类闭集)
    utk_data, utk_labels = load_utk("data/UTKinect_skeletons", "data/actionLabel.txt")
    utk_model = train_model(utk_data, utk_labels, num_classes=10, device=device)
    torch.save(utk_model.state_dict(), "models/utk_model.pth")
