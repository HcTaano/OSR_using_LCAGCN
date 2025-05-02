# ensemble_inference.py

import os
import torch
import numpy as np
from config import train_cfg, path_cfg
from utils import load_msr, load_utk, split_known_unknown
from models import LCAGCN
from openmax_inference import openmax_recalibrate

def ensemble_openmax_inference(models, x, centroids_list, weibulls_list, alpha=5, scale_factor=1.5):
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
            alpha=alpha, scale_factor=scale_factor
        )
        all_probs.append(probs)
    
    # 加权平均，更侧重高置信度预测
    ensemble_probs = np.zeros_like(all_probs[0])
    for probs in all_probs:
        # 计算置信度权重
        confidence = 1.0 - probs[0]  # 非未知类概率和
        ensemble_probs += probs * confidence
    
    return ensemble_probs / np.sum(ensemble_probs)

def main():
    device = train_cfg.DEVICE
    
    # 确定已知类
    known_classes = list(range(18))  # MSR数据集
    
    # 加载数据集
    data, labels = load_msr(path_cfg.msr_data)
    _, _, test_x, test_y, test_ux, test_uy = split_known_unknown(data, labels, known_classes)
    
    # 加载多个模型
    model1 = LCAGCN(num_class=len(known_classes)).to(device)
    model1.load_state_dict(torch.load("checkpoints/model1.pth"))
    model1.eval()
    
    model2 = LCAGCN(num_class=len(known_classes)).to(device)
    model2.load_state_dict(torch.load("checkpoints/model2.pth"))
    model2.eval()
    
    models = [model1, model2]
    
    # 加载对应的Weibull参数
    centroids1 = np.load("weibull/centroids1.npy")
    weibulls1 = torch.load("weibull/weibull1.pth")
    
    centroids2 = np.load("weibull/centroids2.npy")
    weibulls2 = torch.load("weibull/weibull2.pth")
    
    centroids_list = [centroids1, centroids2]
    weibulls_list = [weibulls1, weibulls2]
    
    # 测试数据转换为Tensor
    test_tensor = torch.from_numpy(test_x).to(device)
    
    # 使用集成方法进行推理
    print("对已知类测试样本进行集成推理...")
    ensemble_results = []
    for i in range(len(test_x)):
        x = test_tensor[i:i+1]  # 单样本
        probs = ensemble_openmax_inference(models, x, centroids_list, weibulls_list)
        ensemble_results.append(probs)
    
    # 计算准确率
    correct = 0
    for i, probs in enumerate(ensemble_results):
        if np.argmax(probs[1:]) == test_y[i]:  # 忽略未知类预测
            correct += 1
    
    acc = correct / len(test_y)
    print(f"集成模型在已知类上的准确率: {acc:.4f}")
    
    # 如果有未知类样本，也进行评估
    if len(test_ux) > 0:
        test_u_tensor = torch.from_numpy(test_ux).to(device)
        unknown_results = []
        
        print("对未知类测试样本进行集成推理...")
        for i in range(len(test_ux)):
            x = test_u_tensor[i:i+1]  # 单样本
            probs = ensemble_openmax_inference(models, x, centroids_list, weibulls_list)
            unknown_results.append(probs)
        
        # 计算未知类检测率
        unknown_detect = 0
        for probs in unknown_results:
            if np.argmax(probs) == 0:  # 预测为未知类
                unknown_detect += 1
        
        detect_rate = unknown_detect / len(test_ux)
        print(f"集成模型对未知类的检测率: {detect_rate:.4f}")

if __name__ == "__main__":
    main()