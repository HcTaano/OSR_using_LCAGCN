# 设计文档（design.md）

## 一、项目概览
- **目标**：基于 LC-AGCN 实现骨架动作的开集识别，结合 OpenMax 评估未知类拒识性能。
- **目录结构**：
```
LC-AGCN/ 
 ├─ config.py # 全局配置 
 ├─ train_closedset.py # 闭集训练入口（支持 resume） 
 ├─ extract_activations_and_fit.py # 激活提取 & Weibull 拟合 
 ├─ openmax_inference.py # 开集推理 & AUC 评估 
 ├─ utils/ # 数据加载与划分模块
 ├─ models/ # LC-AGCN 模型定义 
 ├─ checkpoints/ # 检查点与模型权重 
 ├─ weibull/ # Weibull 参数存储 
 └─ outputs/ # 结果输出与可视化
```

## 二、模块功能

| 文件/模块                                  | 功能                                     |
|----------------------------------------|----------------------------------------|
| `config.py`                            | 集中管理超参、文件路径与环境设置                       |
| `utils/data_utils.py`                  | `load_msr`、`load_utk`：加载并预处理两个数据集      |
| `utils/split_known_unknown.py`         | `split_known_unknown`：按已知/未知类划分训练/测试数据 |
| `models/lc_agcn.py`                    | LC-AGCN 网络结构实现                         |
| `train_closedset.py`                   | 闭集训练入口，支持从 checkpoint 恢复               |
| `extract_activations_and_fit.py`       | 提取 penultimate 层特征并对各类距离质心做 Weibull 拟合 |
| `openmax_inference.py`                 | 使用 OpenMax 重校准 logits，并计算开集 AUC/OSCR   |
| `run_pipeline.ps1` / `run_pipeline.sh` | 一键执行闭集训练→Weibull 拟合→开集评估全流程            |

## 三、使用指南

1. **安装环境**  
 ```bash
 conda env create -f environment.yml
 conda activate openmax-lcagcn
```

2. **闭集训练（可 resume）**  
```bash
python train_closedset_noresume.py \
  --dataset MSR \
  --known_classes 0 1 … 14 \
  [--resume checkpoints/msr_checkpoint_epoch14.pth]
```

3. **激活提取 & Weibull 拟合**  
```bash
python extract_activations_and_fit.py
```

4. **OpenMax 开集评估**
```bash
python openmax_inference.py
```

5. **查看结果**
AUC/OSCR 曲线图位于 `outputs/figures/`
数值结果在 `outputs/oscr_results.json`

---

下面是一套在 PowerShell 中从“闭集训练”到“开集评估”的完整命令序列。首先确保已完成 `conda init powershell`，并在新开 PowerShell 中运行以下指令：

0. （仅首次）初始化 PowerShell 对 Conda 的支持 
如果尚未执行过，请先在 Administrator 模式下运行：
```powershell
conda init powershell
```
使 PowerShell 识别 conda activate ,然后重启 PowerShell 窗口

1. 激活 Conda 环境
```powershell
 conda activate lcagcn_osr310
 ```
切换到您用于本项目的环境

2. 闭集训练（MSR 数据集示例）
训练前可选地指定 --resume 路径继续训练
```powershell
python .\train_closedset.py --dataset MSR `
    --known_classes 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 `
    --epochs 1000 `
    --lr 0.0001 `
    --resume checkpoints/msr_checkpoint_epoch99.pth
```
```powershell
python .\train_closedset.py --dataset MSR --known_classes 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 --epochs 100 lr 0.001
```
说明：用 python 命令运行脚本，Windows 下可直接 .\script.py ([How to Run Your Python Scripts and Code](https://realpython.com/run-python-scripts/?utm_source=chatgpt.com))

3. 提取激活并拟合 Weibull 
```powershell
python .\extract_activations_and_fit.py
```
自动加载最新 checkpoint 并生成 weibull 参数

4. OpenMax 开集推理与评估 
```powershell
python .\openmax_inference.py
```
输出开集 AUC/OSCR，并将图保存在 outputs/figures

5. 查看输出结果 
在 PowerShell 中打开结果目录，检查 JSON 与图像文件
```powershell
ii .\outputs\oscr_results.json                                   # ii 是 PowerShell 别名，等同于 Invoke-Item
ii .\outputs\figures\roc_curve_msr.png
```

6. 退出环境
```powershell
conda deactivate                                                 # 回到 base 或退出 conda 环境
```

每行作用如下：

| 步骤 | 命令 | 说明 |
|:----:|:-----|:-----|
| 0 | `conda init powershell` | 配置 PowerShell 启动脚本，使其支持 `conda activate`  |
| 1 | `conda activate lcagcn_osr310` | 切换至项目专用环境（安装了 torch、scipy 等依赖） |
| 2 | `python .\train_closedset.py …` | 执行闭集训练，生成含阶段 checkpoint 与最终模型权重 ([How to Run Your Python Scripts and Code](https://realpython.com/run-python-scripts/?utm_source=chatgpt.com)) |
| 3 | `python .\extract_activations_and_fit.py` | 提取 penultimate 特征并做 Weibull 拟合，保存至 weibull/ 目录 |
| 4 | `python .\openmax_inference.py` | 基于 OpenMax 进行开集推理，报告 AUC/OSCR 并输出图形 |
| 5 | `ii …` | PowerShell 快捷命令打开文件或文件夹（Invoke-Item） |
| 6 | `conda deactivate` | 退出环境，恢复至 base 或系统 shell |

这样，您即可在一个 PowerShell 窗口中顺序完成从数据加载、闭集训练，到 Weibull 拟合、开集评估与结果可视化的全流程。