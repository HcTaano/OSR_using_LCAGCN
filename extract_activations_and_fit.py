# extract_activations_and_fit.py

import os
import torch
import numpy as np
from scipy.stats import exponweib
from utils import load_msr, load_utk, split_known_unknown
from models import LCAGCN

def fit_weibull(dist, tailsize=10):
    tail = np.sort(dist)[-tailsize:]
    params = exponweib.fit(tail, floc=0, f0=1)
    return {'c':params[1], 'scale':params[3], 'min_val':tail[0]}

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for ds, loader in [('MSR', load_msr), ('UTK', load_utk)]:
        # --- 1. 准备模型与 checkpoint 路径 ---
        known = list(range(18)) if ds=='MSR' else list(range(8))
        model = LCAGCN(num_class=len(known)).to(device)

        # 优先尝试加载最终模型权重文件
        ckpt_model = f"checkpoints/{ds.lower()}_model_known{len(known)}.pth"
        if os.path.exists(ckpt_model):
            print(f"Loading model weights from {ckpt_model}")  # :contentReference[oaicite:1]{index=1}
            model.load_state_dict(torch.load(ckpt_model, map_location=device))
        else:
            # 如果最终模型不存在，则尝试最新 checkpoint
            # 查找所有 checkpoint_epoch 文件
            ckpt_dir = "checkpoints"
            pattern = f"{ds.lower()}_checkpoint_epoch"
            cands = [f for f in os.listdir(ckpt_dir) if f.startswith(pattern)]
            if not cands:
                print(f"[WARN] 未找到任何 {ds} checkpoint，跳过 {ds} 部分")
                continue  # 跳过 UTK 或 MSR
            # 选最大 epoch 的文件
            latest = sorted(cands, key=lambda x: int(x[len(pattern):-4]))[-1]
            ckpt_file = os.path.join(ckpt_dir, latest)
            print(f"Loading checkpoint {ckpt_file}")  # :contentReference[oaicite:2]{index=2}
            ckpt = torch.load(ckpt_file, map_location=device)
            model.load_state_dict(ckpt['model_state'])

        model.eval()

        # --- 2. 加载 & 划分数据 ---
        if ds=='MSR':
            data, labels = loader("data/MSRAction3DSkeletonReal3D")
        else:
            data, labels = loader("data/UTKinect_skeletons", "data/UTKinect_skeletons/actionLabel.txt")
        train_x, train_y, *_ = split_known_unknown(data, labels, known)

        # --- 3. 提取特征 & 拟合 Weibull ---
        with torch.no_grad():
            X = torch.from_numpy(train_x).to(device)
            _, feats = model(X)
        feats = feats.cpu().numpy()

        centroids, weibulls = [], []
        for c in known:
            fc = feats[train_y==c]
            mu = fc.mean(axis=0); centroids.append(mu)
            d = np.linalg.norm(fc - mu, axis=1)
            weibulls.append(fit_weibull(d))

        os.makedirs("weibull", exist_ok=True)
        np.save(f"weibull/{ds.lower()}_centroids.npy", np.stack(centroids))
        torch.save(weibulls, f"weibull/{ds.lower()}_weibull.pth")
        print(f"{ds} Weibull 拟合完成。")

if __name__ == "__main__":
    main()


# # extract_activations_and_fit.py
#
# import torch
# import numpy as np
# from scipy.stats import exponweib
# from utils import load_msr, load_utk, split_known_unknown
# from models import LCAGCN
#
# def fit_weibull(dist, tailsize=10):
#     # 对每类激活距离尾部做 Weibull 拟合
#     tail = np.sort(dist)[-tailsize:]
#     params = exponweib.fit(tail, floc=0, f0=1)
#     return {'c':params[1], 'scale':params[3], 'min_val':tail[0]}
#
# def main():
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     for ds, loader in [('MSR', load_msr), ('UTK', load_utk)]:
#         # 加载模型
#         known = list(range(18)) if ds=='MSR' else list(range(8))
#         model = LCAGCN(num_class=len(known)).to(device)
#         model.load_state_dict(torch.load(
#             f"checkpoints/{ds.lower()}_model_known{len(known)}.pth"))
#         model.eval()
#
#         # 加载并划分数据
#         data, labels = (loader("data/MSRAction3DSkeletonReal3D") if ds=='MSR'
#                         else loader("data/UTKinect_skeletons","data/actionLabel.txt"))
#         train_x, train_y, *_ = split_known_unknown(data, labels, known)
#
#         # 提取 penultimate 特征
#         with torch.no_grad():
#             X = torch.tensor(train_x).to(device)
#             _, feats = model(X)
#         feats = feats.cpu().numpy()
#
#         # 每类计算质心并拟合 Weibull
#         centroids, weibulls = [], []
#         for c in known:
#             fc = feats[train_y==c]
#             mu = fc.mean(axis=0); centroids.append(mu)
#             d = np.linalg.norm(fc - mu, axis=1)
#             weibulls.append(fit_weibull(d))
#         np.save(f"weibull/{ds.lower()}_centroids.npy", np.stack(centroids))
#         torch.save(weibulls, f"weibull/{ds.lower()}_weibull.pth")
#         print(f"{ds} Weibull 拟合完成。")
#
# if __name__ == "__main__":
#     main()
#
#
# # # File: extract_activations_and_fit.py
# # import torch, numpy as np
# # from scipy.stats import exponweib
# # from models.lc_agcn import LCAGCN
# # from utils.data_utils import load_msr, load_utk
# #
# # def fit_weibull(distances, tailsize=10):
# #     tail = np.sort(distances)[-tailsize:]
# #     min_val = tail[0]
# #     tail = tail + (10000 - min_val)
# #     params = exponweib.fit(tail, floc=0, f0=1)  # a=1 固定
# #     weibull = {"c": params[1], "scale": params[3], "min_val": min_val}
# #     return weibull
# #
# # def compute_centroids_weibull(features, labels, num_classes, tailsize=10):
# #     centroids = []
# #     w_models = []
# #     for c in range(num_classes):
# #         feats_c = features[labels==c]
# #         centroid = feats_c.mean(axis=0)
# #         centroids.append(centroid)
# #         # 距离到质心
# #         dists = np.linalg.norm(feats_c - centroid, axis=1)
# #         w_models.append(fit_weibull(dists, tailsize))
# #     return np.stack(centroids), w_models
# #
# # if __name__=="__main__":
# #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# #     # MSR 部分：提取激活并拟合 Weibull
# #     msr_data, msr_labels = load_msr("data/MSRAction3DSkeletonReal3D")
# #     model_m = LCAGCN(num_class=20).to(device)
# #     model_m.load_state_dict(torch.load("models/msr_model.pth"))
# #     model_m.eval()
# #     with torch.no_grad():
# #         X = torch.tensor(msr_data).to(device)
# #         logits, feats = model_m(X)
# #     feats = feats.cpu().numpy()
# #     centroids, weibulls = compute_centroids_weibull(feats, msr_labels, num_classes=20, tailsize=10)
# #     np.save("weibull/msr_centroids.npy", centroids)
# #     torch.save(weibulls, "weibull/msr_weibull.pth")
# #     # UTK 部分：同理
# #     utk_data, utk_labels = load_utk("data/UTKinect_skeletons", "data/actionLabel.txt")
# #     model_u = LCAGCN(num_class=10).to(device)
# #     model_u.load_state_dict(torch.load("models/utk_model.pth"))
# #     model_u.eval()
# #     with torch.no_grad():
# #         X = torch.tensor(utk_data).to(device)
# #         logits, feats = model_u(X)
# #     feats = feats.cpu().numpy()
# #     centroids, weibulls = compute_centroids_weibull(feats, utk_labels, num_classes=10, tailsize=10)
# #     np.save("weibull/utk_centroids.npy", centroids)
# #     torch.save(weibulls, "weibull/utk_weibull.pth")
