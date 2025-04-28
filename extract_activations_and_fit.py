# File: extract_activations_and_fit.py
import torch, numpy as np
from scipy.stats import exponweib
from models.lc_agcn import LCAGCN
from utils.data_utils import load_msr, load_utk

def fit_weibull(distances, tailsize=10):
    tail = np.sort(distances)[-tailsize:]
    min_val = tail[0]
    tail = tail + (10000 - min_val)
    params = exponweib.fit(tail, floc=0, f0=1)  # a=1 固定
    weibull = {"c": params[1], "scale": params[3], "min_val": min_val}
    return weibull

def compute_centroids_weibull(features, labels, num_classes, tailsize=10):
    centroids = []
    w_models = []
    for c in range(num_classes):
        feats_c = features[labels==c]
        centroid = feats_c.mean(axis=0)
        centroids.append(centroid)
        # 距离到质心
        dists = np.linalg.norm(feats_c - centroid, axis=1)
        w_models.append(fit_weibull(dists, tailsize))
    return np.stack(centroids), w_models

if __name__=="__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # MSR 部分：提取激活并拟合 Weibull
    msr_data, msr_labels = load_msr("data/MSRAction3DSkeletonReal3D")
    model_m = LCAGCN(num_class=20).to(device)
    model_m.load_state_dict(torch.load("models/msr_model.pth"))
    model_m.eval()
    with torch.no_grad():
        X = torch.tensor(msr_data).to(device)
        logits, feats = model_m(X)
    feats = feats.cpu().numpy()
    centroids, weibulls = compute_centroids_weibull(feats, msr_labels, num_classes=20, tailsize=10)
    np.save("weibull/msr_centroids.npy", centroids)
    torch.save(weibulls, "weibull/msr_weibull.pth")
    # UTK 部分：同理
    utk_data, utk_labels = load_utk("data/UTKinect_skeletons", "data/actionLabel.txt")
    model_u = LCAGCN(num_class=10).to(device)
    model_u.load_state_dict(torch.load("models/utk_model.pth"))
    model_u.eval()
    with torch.no_grad():
        X = torch.tensor(utk_data).to(device)
        logits, feats = model_u(X)
    feats = feats.cpu().numpy()
    centroids, weibulls = compute_centroids_weibull(feats, utk_labels, num_classes=10, tailsize=10)
    np.save("weibull/utk_centroids.npy", centroids)
    torch.save(weibulls, "weibull/utk_weibull.pth")
