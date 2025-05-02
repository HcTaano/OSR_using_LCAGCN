# openmax_inference.py

import os
import argparse
import torch
import numpy as np
from scipy.stats import exponweib
import matplotlib.pyplot as plt

from config import train_cfg, path_cfg
from utils import load_msr, load_utk, split_known_unknown
from models import LCAGCN
from sklearn.metrics import roc_auc_score  # 计算 AUC
from sklearn.metrics import classification_report, confusion_matrix  # 闭集报告&#8203;:contentReference[oaicite:0]{index=0}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--skip-utk', action='store_true',
                   help="跳过 UTK 数据集评估（默认评估 MSR + UTK）")
    p.add_argument('--alpha', type=int, default=3,
                   help="OpenMax重校准时考虑的Top-α类别数量（默认3）")
    p.add_argument('--tailsize', type=int, default=10,
                   help="Weibull拟合时使用的尾部样本数量（默认10）")
    p.add_argument('--visualize', action='store_true',
                   help="是否生成ROC曲线和未知类分数分布图")
    return p.parse_args()

# ```powershell
# python openmax_inference.py --skip-utk --alpha=5 --tailsize=15  --visualize
# ```

def weibull_cdf(x, w):
    """计算经调整的Weibull累积分布函数(CDF)值"""
    return exponweib.cdf(x + (10000 - w['min_val']), a=1, c=w['c'], loc=0, scale=w['scale'])


def openmax_recalibrate(logits, feat, cents, wbls, alpha=5, scale_factor=1.5):
    """
    改进的OpenMax重校准:
    1. 动态alpha选择
    2. 距离缩放优化
    3. 改进的未知样本得分计算

    参数:
        logits: 网络输出的原始logits
        feat: 对应的特征向量
        cents: 各类质心
        wbls: 各类的Weibull参数
        alpha: 考虑重校准的前α个类
        scale_factor: 距离缩放因子，用于调整未知样本识别灵敏度

    返回:
        包含未知类在内的概率分布 (未知类为第0个元素)
    """
    # 动态调整alpha基于logits分布
    max_logit = np.max(logits)
    logit_diff = max_logit - np.mean(logits)
    if logit_diff > 5.0:  # 高置信度，减小alpha
        alpha = max(2, alpha - 1)
    elif logit_diff < 2.0:  # 低置信度，增加alpha
        alpha = min(8, alpha + 1)

    # 选择top-alpha类做激活衰减
    ranked = np.argsort(logits)[::-1]
    omega = np.zeros_like(logits)
    for i in range(min(alpha, len(ranked))):
        # 非线性衰减权重
        omega[ranked[i]] = (1 - 0.8 * (i / alpha) ** 2)

    # 计算到各类质心的距离
    dists = np.linalg.norm(feat - cents, axis=1) * scale_factor

    # 计算Weibull概率
    wprobs = np.array([weibull_cdf(d, w) for d, w in zip(dists, wbls)])

    # 激活衰减
    logits_hat = logits * (1 - omega * wprobs)

    # 改进的未知类得分计算，考虑整体分布
    unk_score = np.sum(logits - logits_hat) * (1.0 + 0.2 * np.std(wprobs))

    # 合并未知类得分和重校准后的各类得分
    new_logits = np.concatenate(([unk_score], logits_hat))

    # 生成概率分布，稍微平滑处理
    exp = np.exp(new_logits - np.max(new_logits))
    return exp / np.sum(exp)

def ensemble_openmax_inference(models, x, centroids_list, weibulls_list):
    """使用多个模型组合预测，提高开集识别可靠性"""
    all_probs = []
    
    # 获取每个模型的预测
    for i, model in enumerate(models):
        with torch.no_grad():
            logits, feats = model(x)
            logits_np = logits.cpu().numpy()
            feats_np = feats.cpu().numpy()
        
        # 应用优化后的OpenMax
        probs = openmax_recalibrate(
            logits_np, feats_np, 
            centroids_list[i], weibulls_list[i],
            alpha=5, scale_factor=1.5
        )
        all_probs.append(probs)
    
    # 加权平均，更侧重高置信度预测
    ensemble_probs = np.zeros_like(all_probs[0])
    for probs in all_probs:
        # 计算置信度权重
        confidence = 1.0 - probs[0]  # 非未知类概率和
        ensemble_probs += probs * confidence
    
    return ensemble_probs / np.sum(ensemble_probs)
    
# def openmax_recalibrate(logits, feat, cents, wbls, alpha=3):
#     """
#     实现OpenMax算法的重校准
#
#     参数:
#         logits: 网络输出的原始logits
#         feat: 对应的特征向量
#         cents: 各类质心
#         wbls: 各类的Weibull参数
#         alpha: 考虑重校准的前α个类
#
#     返回:
#         包含未知类在内的概率分布 (未知类为第0个元素)
#     """
#     # 选择top-alpha类做激活衰减
#     ranked = np.argsort(logits)[::-1]
#     omega = np.zeros_like(logits)
#     for i in range(min(alpha, len(ranked))):
#         # 线性衰减权重
#         omega[ranked[i]] = (alpha - i) / alpha
#
#     # 计算到各类质心的距离
#     dists = np.linalg.norm(feat - cents, axis=1)
#
#     # 计算Weibull概率
#     wprobs = np.array([weibull_cdf(d, w) for d, w in zip(dists, wbls)])
#
#     # 激活衰减
#     logits_hat = logits * (1 - omega * wprobs)
#
#     # 未知类得分：原始激活减去重校准激活的总和
#     unk_score = np.sum(logits - logits_hat)
#
#     # 合并未知类得分和重校准后的各类得分
#     new_logits = np.concatenate(([unk_score], logits_hat))
#
#     # 转为概率分布
#     exp = np.exp(new_logits - np.max(new_logits))
#     return exp / np.sum(exp)

def compute_oscr(y_true, open_scores, preds):
    """
    计算OSCR (Open-Set Classification Rate)
    
    参数:
        y_true: 真实标签
        open_scores: 开集得分
        preds: 闭集预测结果
        
    返回:
        oscr: 开集识别率
    """
    # 创建二分类标签: 1表示已知类预测正确, 0表示已知类预测错误或未知类
    binary_labels = np.zeros_like(y_true, dtype=int)
    for i, (y, pred) in enumerate(zip(y_true, preds)):
        if y == pred:
            binary_labels[i] = 1
    
    # 计算ROC AUC
    # 注意: 使用1-open_scores因为较低的open_score表示更可能是已知类
    return roc_auc_score(binary_labels, 1-open_scores)

def plot_roc_curve(ds, y_true, y_score):
    """绘制ROC曲线并保存"""
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{ds} Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    # 确保输出目录存在
    os.makedirs('outputs/figures', exist_ok=True)
    plt.savefig(f'outputs/figures/roc_curve_{ds.lower()}.png')
    plt.close()

def eval_ds(ds, loader, label_args, args):
    known = list(range(label_args))  # 已知类列表
    
    # 加载模型权重
    fn = f"checkpoints/{ds.lower()}_model_known{len(known)}.pth"
    if not os.path.exists(fn):  # 检查文件存在性
        print(f"[WARN] 未找到 {fn}，跳过 {ds}")
        return
    
    model = LCAGCN(num_class=len(known)).to(train_cfg.DEVICE)
    model.load_state_dict(torch.load(fn, map_location=train_cfg.DEVICE))
    model.eval()

    # 加载并划分数据
    if ds=='MSR':
        data, labels = load_msr(path_cfg.msr_data)
    else:
        data, labels = load_utk(path_cfg.utk_data, path_cfg.utk_label)
    
    train_x, train_y, test_x, test_y, test_ux, test_uy = split_known_unknown(data, labels, known)

    # 加载质心和Weibull参数
    centroids_file = f"weibull/{ds.lower()}_centroids.npy"
    weibull_file = f"weibull/{ds.lower()}_weibull.pth"
    
    if not (os.path.exists(centroids_file) and os.path.exists(weibull_file)):
        print(f"[ERROR] 未找到Weibull参数文件，请先运行extract_activations_and_fit.py")
        return
    
    centroids = np.load(centroids_file)
    weibulls = torch.load(weibull_file)
    
    print(f"\n=== {ds} 数据集评估 ===")
    print(f"已知类数量: {len(known)}")
    print(f"已知测试样本数: {len(test_y)}")
    print(f"未知测试样本数: {len(test_uy)}")
    
    # 获取特征和logits
    with torch.no_grad():
        # 已知类测试样本
        logits, feats = model(torch.from_numpy(test_x).to(train_cfg.DEVICE))
        logits_np = logits.cpu().numpy()
        feats_np = feats.cpu().numpy()
        preds = torch.argmax(logits, axis=1).cpu().numpy()
        
        # 未知类测试样本
        if len(test_ux) > 0:
            u_logits, u_feats = model(torch.from_numpy(test_ux).to(train_cfg.DEVICE))
            u_logits_np = u_logits.cpu().numpy()
            u_feats_np = u_feats.cpu().numpy()
    
    # 计算OpenMax得分
    print("计算OpenMax得分...")
    open_scores = []  # 未知类概率
    openmax_preds = []  # OpenMax预测的类别
    
    for i in range(len(test_x)):
        probs = openmax_recalibrate(
            logits_np[i], feats_np[i], centroids, weibulls, alpha=args.alpha
        )
        open_scores.append(probs[0])  # 第0个元素是未知类的概率
        
        # 如果未知类概率最大，预测为-1，否则取已知类中概率最大的
        if np.argmax(probs) == 0:
            pred_class = -1  # 未知类
        else:
            pred_class = np.argmax(probs[1:])  # 找出概率最大的已知类
        
        openmax_preds.append(pred_class)
    
    # 未知类样本的开集得分
    unknown_open_scores = []
    if len(test_ux) > 0:
        for i in range(len(test_ux)):
            u_probs = openmax_recalibrate(
                u_logits_np[i], u_feats_np[i], centroids, weibulls, alpha=args.alpha
            )
            unknown_open_scores.append(u_probs[0])
    
    # 计算开集AUC
    if len(test_ux) > 0:
        # 准备已知/未知二分类标签和得分
        binary_y = np.concatenate([np.zeros(len(test_x)), np.ones(len(test_ux))])
        all_scores = np.concatenate([open_scores, unknown_open_scores])
        
        auc = roc_auc_score(binary_y, all_scores)
        print(f"{ds} 开集AUC: {auc:.3f}")
        
        # 绘制ROC曲线
        if args.visualize:
            plot_roc_curve(ds, binary_y, all_scores)
    else:
        print(f"[WARN] {ds} 没有未知类样本，无法计算开集AUC")
    
    # 闭集预测与评估
    print(f"\n{ds} 闭集 Classification Report：")
    print(classification_report(test_y, preds, zero_division=0))
    print(f"{ds} 混淆矩阵：\n", confusion_matrix(test_y, preds))
    
    # 标准闭集准确率
    closed_acc = np.mean(preds == test_y)
    print(f"{ds} 闭集准确率: {closed_acc:.3f}")
    
    # OpenMax处理后的已知类准确率
    openmax_valid_idx = [i for i, p in enumerate(openmax_preds) if p != -1]
    if openmax_valid_idx:
        openmax_acc = np.mean([openmax_preds[i] == test_y[i] for i in openmax_valid_idx])
        print(f"{ds} OpenMax已知类准确率: {openmax_acc:.3f}")
    else:
        openmax_acc = 0
        print(f"{ds} OpenMax拒绝了所有已知类样本")
    
    # 计算已知类的拒识率
    rejection_rate = np.mean([1 if p == -1 else 0 for p in openmax_preds])
    print(f"{ds} 已知类拒识率: {rejection_rate:.3f}")
    
    # 计算未知类的拒识率（如果有未知类样本）
    if len(test_ux) > 0:
        unknown_rejection_threshold = 0.5  # 未知类概率阈值
        unknown_rejection_rate = np.mean([1 if p > unknown_rejection_threshold else 0 for p in unknown_open_scores])
        print(f"{ds} 未知类正确拒识率(阈值{unknown_rejection_threshold}): {unknown_rejection_rate:.3f}")
    
    # 计算OSCR
    if len(test_ux) > 0:
        # 合并已知类和未知类
        all_labels = np.concatenate([test_y, np.ones_like(test_uy) * -1])  # -1表示未知类
        all_preds = np.concatenate([preds, np.zeros_like(test_uy)])  # 未知类的预测无意义
        all_open_scores = np.concatenate([open_scores, unknown_open_scores])
        
        oscr = compute_oscr(all_labels, all_open_scores, all_preds)
        print(f"{ds} OSCR: {oscr:.3f}")
    else:
        # 只有已知类时的OSCR
        oscr = compute_oscr(test_y, open_scores, preds)
        print(f"{ds} OSCR (仅已知类): {oscr:.3f}")
    
    # 保存评估结果
    os.makedirs('outputs', exist_ok=True)
    results = {
        'dataset': ds,
        'num_known_classes': len(known),
        'num_test_known': len(test_y),
        'num_test_unknown': len(test_uy),
        'closed_set_accuracy': float(closed_acc),
        'openmax_known_accuracy': float(openmax_acc),
        'known_rejection_rate': float(rejection_rate),
        'oscr': float(oscr)
    }
    
    if len(test_ux) > 0:
        results['unknown_rejection_rate'] = float(unknown_rejection_rate)
        results['open_set_auc'] = float(auc)
    
    # 可视化未知类分数分布（如果有可视化标志）
    if args.visualize and len(test_ux) > 0:
        plt.figure(figsize=(10, 6))
        plt.hist(open_scores, bins=30, alpha=0.5, label='已知类')
        plt.hist(unknown_open_scores, bins=30, alpha=0.5, label='未知类')
        plt.xlabel('未知类概率')
        plt.ylabel('样本数量')
        plt.title(f'{ds} 已知类vs未知类的未知类概率分布')
        plt.legend()
        plt.savefig(f'outputs/figures/{ds.lower()}_unknown_score_dist.png')
        plt.close()
    
    return results

if __name__=="__main__":
    args = parse_args()
    
    # 确保输出目录存在
    os.makedirs('outputs', exist_ok=True)
    if args.visualize:
        os.makedirs('outputs/figures', exist_ok=True)
    
    results = {}
    
    # MSR 评估
    msr_results = eval_ds('MSR', load_msr, 18, args)
    if msr_results:
        results['MSR'] = msr_results
    
    # UTK 评估（可跳过）
    if not args.skip_utk:
        utk_results = eval_ds('UTK', load_utk, 8, args)
        if utk_results:
            results['UTK'] = utk_results
    
    # 保存所有结果
    import json
    with open('outputs/oscr_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n评估完成！结果已保存到 outputs/oscr_results.json")


# # openmax_inference.py
#
# import torch
# import numpy as np
# from scipy.stats import exponweib
# from sklearn.metrics import roc_auc_score
# from utils import load_msr, load_utk, split_known_unknown
# from models import LCAGCN
#
# def weibull_cdf(x, w):
#     # 计算经调整的 Weibull CDF
#     return exponweib.cdf(x + (10000-w['min_val']), a=1, c=w['c'], loc=0, scale=w['scale'])
#
# def openmax_recalibrate(logits, feat, cents, wbls, alpha=3):
#     # 选 top-alpha 类做衰减
#     ranked = np.argsort(logits)[::-1]
#     omega = np.zeros_like(logits)
#     for i in range(alpha):
#         omega[ranked[i]] = (alpha - i) / alpha
#     dists = np.linalg.norm(feat - cents, axis=1)
#     wprobs = np.array([weibull_cdf(d, w) for d,w in zip(dists, wbls)])
#     logits_hat = logits * (1 - omega * wprobs)
#     unk_score = (logits - logits_hat).sum()
#     new = np.concatenate(([unk_score], logits_hat))
#     exp = np.exp(new - new.max()); return exp/exp.sum()
#
# def eval_ds(ds, known):
#     # 加载 & 划分
#     data, labels = (load_msr("data/MSRAction3DSkeletonReal3D") if ds=='MSR'
#                     else load_utk("data/UTKinect_skeletons", "data/UTKinect_skeletons/actionLabel.txt"))
#     _, _, tk_x, tk_y, tu_x, tu_y = split_known_unknown(data, labels, known)
#
#     # 模型与 Weibull 参数
#     model = LCAGCN(num_class=len(known))
#     model.load_state_dict(torch.load(f"checkpoints/{ds.lower()}_model_known{len(known)}.pth"))
#     model.eval()
#     cents = np.load(f"weibull/{ds.lower()}_centroids.npy")
#     wbls = torch.load(f"weibull/{ds.lower()}_weibull.pth")
#
#     # 前向
#     with torch.no_grad():
#         lk, fk = model(torch.tensor(tk_x)); lu, fu = model(torch.tensor(tu_x))
#     logits = np.vstack((lk.numpy(), lu.numpy()))
#     feats  = np.vstack((fk.numpy(), fu.numpy()))
#
#     # 计算 Open-set AUC
#     y_true, y_score = [], []
#     for lbl, log, feat in zip(np.r_[tk_y, tu_y], logits, feats):
#         p = openmax_recalibrate(log, feat, cents, wbls)
#         y_true.append(int(lbl not in known))
#         y_score.append(p[0])
#     auc = roc_auc_score(y_true, y_score)
#     print(f"{ds} 开集 AUC: {auc:.3f}")
#
# if __name__ == "__main__":
#     eval_ds('MSR', list(range(18)))
#     eval_ds('UTK', list(range(8)))
#
#
#
#
# # # File: openmax_inference.py
# # import torch, numpy as np
# # from scipy.stats import exponweib
# # from models.lc_agcn import LCAGCN
# # from utils.data_utils import load_msr, load_utk
# #
# # def w_score(dist, weibull):
# #     # Weibull CDF 得分
# #     c = weibull["c"]; scale = weibull["scale"]; min_val = weibull["min_val"]
# #     vals = dist + (10000 - min_val)
# #     return exponweib.cdf(vals, a=1, c=c, loc=0, scale=scale)
# #
# # def openmax_recalibrate(logits, emb, centroids, weibulls, alpha=3):
# #     # logits, emb: single样本 (np.array)
# #     # centroids: [K, dim], weibulls: list of dict, alpha 取 top-alpha 类
# #     cls_idxs = np.argsort(logits)[::-1]
# #     alpha_coeffs = np.zeros_like(logits)
# #     for i in range(alpha):
# #         alpha_coeffs[cls_idxs[i]] = (alpha - i) / alpha
# #     # 计算到各类质心距离
# #     dists = np.linalg.norm(emb - centroids, axis=1)
# #     weibull_probs = np.array([w_score(dists[i], weibulls[i]) for i in range(len(weibulls))])
# #     # 重标 logits
# #     logits_hat = logits * (1 - alpha_coeffs * weibull_probs)
# #     logit_unk = (logits - logits_hat).sum()
# #     new_logits = np.concatenate(([logit_unk], logits_hat))
# #     # 返回 SoftMax 概率 (0: unknown)
# #     exp = np.exp(new_logits - np.max(new_logits))
# #     probs = exp / exp.sum()
# #     return probs
# #
# # if __name__=="__main__":
# #     # 评估 MSR 开集
# #     msr_data, msr_labels = load_msr("data/MSRAction3DSkeletonReal3D")
# #     model_m = LCAGCN(num_class=20)
# #     model_m.load_state_dict(torch.load("models/msr_model.pth"))
# #     model_m.eval()
# #     centroids = np.load("weibull/msr_centroids.npy")
# #     weibulls = torch.load("weibull/msr_weibull.pth")
# #     known_classes = set(range(15))  # 假设前15类为已知，其余5类为未知
# #     y_true_open, y_score_open = [], []
# #     correct = 0; total_known = 0
# #     with torch.no_grad():
# #         X = torch.tensor(msr_data)
# #         logits_batch, feats_batch = model_m(X)
# #         logits_np = logits_batch.numpy(); feats_np = feats_batch.numpy()
# #     for i in range(len(msr_labels)):
# #         label = msr_labels[i]
# #         probs = openmax_recalibrate(logits_np[i], feats_np[i], centroids, weibulls, alpha=3)
# #         # 闭集分类准确率（仅对已知类计数）
# #         if label in known_classes:
# #             total_known += 1
# #             pred = np.argmax(probs[1:])  # 忽略未知分量，对已知类取最大
# #             if pred == label: correct += 1
# #         # 开集未知概率得分和标签（未知为1）
# #         is_unknown = 1 if label not in known_classes else 0
# #         y_true_open.append(is_unknown)
# #         y_score_open.append(probs[0])  # 概率未知类分量
# #     closed_acc = correct / total_known
# #     from sklearn.metrics import roc_auc_score
# #     auc = roc_auc_score(y_true_open, y_score_open)
# #     print(f"MSR Closed-set Acc: {closed_acc:.3f}, Open-set AUC: {auc:.3f}")
# #
# #     # 评估 UTKinect 开集（示例: 前8类为已知）
# #     utk_data, utk_labels = load_utk("data/UTKinect_skeletons", "data/actionLabel.txt")
# #     model_u = LCAGCN(num_class=10)
# #     model_u.load_state_dict(torch.load("models/utk_model.pth"))
# #     model_u.eval()
# #     centroids = np.load("weibull/utk_centroids.npy")
# #     weibulls = torch.load("weibull/utk_weibull.pth")
# #     known_classes = set(range(8))
# #     y_true_open, y_score_open = [], []
# #     correct = 0; total_known = 0
# #     with torch.no_grad():
# #         X = torch.tensor(utk_data)
# #         logits_batch, feats_batch = model_u(X)
# #         logits_np = logits_batch.numpy(); feats_np = feats_batch.numpy()
# #     for i in range(len(utk_labels)):
# #         label = utk_labels[i]
# #         probs = openmax_recalibrate(logits_np[i], feats_np[i], centroids, weibulls, alpha=3)
# #         if label in known_classes:
# #             total_known += 1
# #             pred = np.argmax(probs[1:])
# #             if pred == label: correct += 1
# #         y_true_open.append(1 if label not in known_classes else 0)
# #         y_score_open.append(probs[0])
# #     closed_acc = correct / total_known
# #     auc = roc_auc_score(y_true_open, y_score_open)
# #     print(f"UTK Closed-set Acc: {closed_acc:.3f}, Open-set AUC: {auc:.3f}")
