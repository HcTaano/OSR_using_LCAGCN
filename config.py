# config.py

import os
from dataclasses import dataclass, field  # 使用 field 以支持可变默认值
from typing import List

@dataclass
class BaseConfig:
    # 通用设置
    T: int = 20               # 序列长度（帧数）
    V: int = 20               # 关节数
    DEVICE: str = 'cuda' if os.getenv('CUDA')=='1' else 'cpu'

@dataclass
class TrainConfig(BaseConfig):
    # 训练超参
    dataset: str = 'MSR'      # 'MSR' 或 'UTK'
    # 将可变默认 list 改为 default_factory 以避免 dataclass 报错
    known_classes: List[int] = field(default_factory=lambda: list(range(18)))  # 扩大到 18 类
    batch_size: int = 128     # 批大小
    epochs: int = 1000        # 总训练轮次
    lr: float = 1e-4          # 初始学习率
    resume: str = None        # checkpoint 路径

@dataclass
class PathsConfig(BaseConfig):
    # 路径设置
    msr_data: str = 'data/MSRAction3DSkeletonReal3D'
    utk_data: str = 'data/UTKinect_skeletons/joints'
    utk_label: str = 'data/UTKinect_skeletons/actionLabel.txt'
    ckpt_dir: str = 'checkpoints'
    weibull_dir: str = 'weibull'
    outputs_dir: str = 'outputs'

# 实例化供全局 import
train_cfg = TrainConfig()
path_cfg = PathsConfig()


# # config.py
#
# import os
# from dataclasses import dataclass, field
# from typing import List
#
# @dataclass
# class BaseConfig:
#     # 通用设置
#     T: int = 20               # 序列长度
#     V: int = 20               # 关节数
#     DEVICE: str = 'cuda' if os.getenv('CUDA')=='1' else 'cpu'
#
# @dataclass
# class TrainConfig(BaseConfig):
#     # 训练超参
#     dataset: str = 'MSR'      # 'MSR' 或 'UTK'
#     # 可变默认值必须用 default_factory 才合法，否则抛 ValueError :contentReference[oaicite:0]{index=0}
#     known_classes: List[int] = field(default_factory=lambda: list(range(15)))
#     batch_size: int = 128
#     epochs: int = 500
#     lr: float = 1e-4
#     resume: str = None        # checkpoint 路径
#
# @dataclass
# class PathsConfig(BaseConfig):
#     # 路径设置
#     msr_data: str = 'data/MSRAction3DSkeletonReal3D'
#     utk_data: str = 'data/UTKinect_skeletons/joints'
#     utk_label: str = 'data/UTKinect_skeletons/actionLabel.txt'
#     ckpt_dir: str = 'checkpoints'
#     weibull_dir: str = 'weibull'
#     outputs_dir: str = 'outputs'
#
# # 实例化供全局 import
# train_cfg = TrainConfig()
# path_cfg = PathsConfig()
#
#
# # # config.py
# #
# # import os
# # from dataclasses import dataclass
# # from typing import List
# #
# # @dataclass
# # class BaseConfig:
# #     # 通用设置
# #     T: int = 20               # 序列长度
# #     V: int = 20               # 关节数
# #     DEVICE: str = 'cuda' if os.getenv('CUDA')=='1' else 'cpu'
# #
# # @dataclass
# # class TrainConfig(BaseConfig):
# #     # 训练超参
# #     dataset: str = 'MSR'      # 'MSR' 或 'UTK'
# #     known_classes: List[int] = (list(range(15)))  # MSR 默认前15类
# #     batch_size: int = 64
# #     epochs: int = 100
# #     lr: float = 1e-3
# #     resume: str = None        # checkpoint 路径
# #
# # @dataclass
# # class PathsConfig(BaseConfig):
# #     # 路径设置
# #     msr_data: str = 'data/MSRAction3DSkeletonReal3D'
# #     utk_data: str = 'data/UTKinect_skeletons/joints'
# #     utk_label: str = 'data/UTKinect_skeletons/actionLabel.txt'
# #     ckpt_dir: str = 'checkpoints'
# #     weibull_dir: str = 'weibull'
# #     outputs_dir: str = 'outputs'
# #
# # # 实例化供全局 import
# # train_cfg = TrainConfig()
# # path_cfg = PathsConfig()
