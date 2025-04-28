# openmax_inference.py

import torch
import numpy as np
from scipy.stats import exponweib
from sklearn.metrics import roc_auc_score
from utils import load_msr, load_utk, split_known_unknown
from models import LCAGCN

def weibull_cdf(x, w):
    # 计算经调整的 Weibull CDF
    return exponweib.cdf(x + (10000-w['min_val']), a=1, c=w['c'], loc=0, scale=w['scale'])

def openmax_recalibrate(logits, feat, cents, wbls, alpha=3):
    # 选 top-alpha 类做衰减
    ranked = np.argsort(logits)[::-1]
    omega = np.zeros_like(logits)
    for i in range(alpha):
        omega[ranked[i]] = (alpha - i) / alpha
    dists = np.linalg.norm(feat - cents, axis=1)
    wprobs = np.array([weibull_cdf(d, w) for d,w in zip(dists, wbls)])
    logits_hat = logits * (1 - omega * wprobs)
    unk_score = (logits - logits_hat).sum()
    new = np.concatenate(([unk_score], logits_hat))
    exp = np.exp(new - new.max()); return exp/exp.sum()

def eval_ds(ds, known):
    # 加载 & 划分
    data, labels = (load_msr("data/MSRAction3DSkeletonReal3D") if ds=='MSR'
                    else load_utk("data/UTKinect_skeletons", "data/UTKinect_skeletons/actionLabel.txt"))
    _, _, tk_x, tk_y, tu_x, tu_y = split_known_unknown(data, labels, known)

    # 模型与 Weibull 参数
    model = LCAGCN(num_class=len(known))
    model.load_state_dict(torch.load(f"checkpoints/{ds.lower()}_model_known{len(known)}.pth"))
    model.eval()
    cents = np.load(f"weibull/{ds.lower()}_centroids.npy")
    wbls = torch.load(f"weibull/{ds.lower()}_weibull.pth")

    # 前向
    with torch.no_grad():
        lk, fk = model(torch.tensor(tk_x)); lu, fu = model(torch.tensor(tu_x))
    logits = np.vstack((lk.numpy(), lu.numpy()))
    feats  = np.vstack((fk.numpy(), fu.numpy()))

    # 计算 Open-set AUC
    y_true, y_score = [], []
    for lbl, log, feat in zip(np.r_[tk_y, tu_y], logits, feats):
        p = openmax_recalibrate(log, feat, cents, wbls)
        y_true.append(int(lbl not in known))
        y_score.append(p[0])
    auc = roc_auc_score(y_true, y_score)
    print(f"{ds} 开集 AUC: {auc:.3f}")

if __name__ == "__main__":
    eval_ds('MSR', list(range(15)))
    eval_ds('UTK', list(range(8)))




# # File: openmax_inference.py
# import torch, numpy as np
# from scipy.stats import exponweib
# from models.lc_agcn import LCAGCN
# from utils.data_utils import load_msr, load_utk
#
# def w_score(dist, weibull):
#     # Weibull CDF 得分
#     c = weibull["c"]; scale = weibull["scale"]; min_val = weibull["min_val"]
#     vals = dist + (10000 - min_val)
#     return exponweib.cdf(vals, a=1, c=c, loc=0, scale=scale)
#
# def openmax_recalibrate(logits, emb, centroids, weibulls, alpha=3):
#     # logits, emb: single样本 (np.array)
#     # centroids: [K, dim], weibulls: list of dict, alpha 取 top-alpha 类
#     cls_idxs = np.argsort(logits)[::-1]
#     alpha_coeffs = np.zeros_like(logits)
#     for i in range(alpha):
#         alpha_coeffs[cls_idxs[i]] = (alpha - i) / alpha
#     # 计算到各类质心距离
#     dists = np.linalg.norm(emb - centroids, axis=1)
#     weibull_probs = np.array([w_score(dists[i], weibulls[i]) for i in range(len(weibulls))])
#     # 重标 logits
#     logits_hat = logits * (1 - alpha_coeffs * weibull_probs)
#     logit_unk = (logits - logits_hat).sum()
#     new_logits = np.concatenate(([logit_unk], logits_hat))
#     # 返回 SoftMax 概率 (0: unknown)
#     exp = np.exp(new_logits - np.max(new_logits))
#     probs = exp / exp.sum()
#     return probs
#
# if __name__=="__main__":
#     # 评估 MSR 开集
#     msr_data, msr_labels = load_msr("data/MSRAction3DSkeletonReal3D")
#     model_m = LCAGCN(num_class=20)
#     model_m.load_state_dict(torch.load("models/msr_model.pth"))
#     model_m.eval()
#     centroids = np.load("weibull/msr_centroids.npy")
#     weibulls = torch.load("weibull/msr_weibull.pth")
#     known_classes = set(range(15))  # 假设前15类为已知，其余5类为未知
#     y_true_open, y_score_open = [], []
#     correct = 0; total_known = 0
#     with torch.no_grad():
#         X = torch.tensor(msr_data)
#         logits_batch, feats_batch = model_m(X)
#         logits_np = logits_batch.numpy(); feats_np = feats_batch.numpy()
#     for i in range(len(msr_labels)):
#         label = msr_labels[i]
#         probs = openmax_recalibrate(logits_np[i], feats_np[i], centroids, weibulls, alpha=3)
#         # 闭集分类准确率（仅对已知类计数）
#         if label in known_classes:
#             total_known += 1
#             pred = np.argmax(probs[1:])  # 忽略未知分量，对已知类取最大
#             if pred == label: correct += 1
#         # 开集未知概率得分和标签（未知为1）
#         is_unknown = 1 if label not in known_classes else 0
#         y_true_open.append(is_unknown)
#         y_score_open.append(probs[0])  # 概率未知类分量
#     closed_acc = correct / total_known
#     from sklearn.metrics import roc_auc_score
#     auc = roc_auc_score(y_true_open, y_score_open)
#     print(f"MSR Closed-set Acc: {closed_acc:.3f}, Open-set AUC: {auc:.3f}")
#
#     # 评估 UTKinect 开集（示例: 前8类为已知）
#     utk_data, utk_labels = load_utk("data/UTKinect_skeletons", "data/actionLabel.txt")
#     model_u = LCAGCN(num_class=10)
#     model_u.load_state_dict(torch.load("models/utk_model.pth"))
#     model_u.eval()
#     centroids = np.load("weibull/utk_centroids.npy")
#     weibulls = torch.load("weibull/utk_weibull.pth")
#     known_classes = set(range(8))
#     y_true_open, y_score_open = [], []
#     correct = 0; total_known = 0
#     with torch.no_grad():
#         X = torch.tensor(utk_data)
#         logits_batch, feats_batch = model_u(X)
#         logits_np = logits_batch.numpy(); feats_np = feats_batch.numpy()
#     for i in range(len(utk_labels)):
#         label = utk_labels[i]
#         probs = openmax_recalibrate(logits_np[i], feats_np[i], centroids, weibulls, alpha=3)
#         if label in known_classes:
#             total_known += 1
#             pred = np.argmax(probs[1:])
#             if pred == label: correct += 1
#         y_true_open.append(1 if label not in known_classes else 0)
#         y_score_open.append(probs[0])
#     closed_acc = correct / total_known
#     auc = roc_auc_score(y_true_open, y_score_open)
#     print(f"UTK Closed-set Acc: {closed_acc:.3f}, Open-set AUC: {auc:.3f}")
