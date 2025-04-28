# 下载 MSRAction3D 骨架数据
```
Invoke-WebRequest "https://github.com/ahmedius2/Human-Action-Recognition-using-Dense-LSTM/raw/refs/heads/master/MSRAction3DSkeletonReal3D.tar.gz" -OutFile "MSR3D.tar.gz"
tar -xzf MSR3D.tar.gz -C .\data\MSRAction3DSkeletonReal3D
```

# 下载 UTKinect 骨架数据和标签
```powershell
Invoke-WebRequest "https://cvrc.ece.utexas.edu/KinectDatasets/joints.zip" -OutFile "joints.zip"
Expand-Archive .\joints.zip -DestinationPath .\data\UTKinect_skeletons
Invoke-WebRequest "https://cvrc.ece.utexas.edu/KinectDatasets/actionLabel.txt" -OutFile ".\data\actionLabel.txt"
```

# 训练 LC-AGCN（闭集）
```powershell
python train_closedset.py
```

# 提取训练激活并拟合 Weibull
```powershell
python extract_activations_and_fit.py
```

# OpenMax 开集推理并输出结果（含闭集准确率和开集 AUC）
```powershell
python openmax_inference.py
```

# 生成 OSCR 曲线可视化
# （可在 openmax_inference.py 中加入计算 OSCR 并保存图像的代码）
