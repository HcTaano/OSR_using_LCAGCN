# utils/split_known_unknown.py

import numpy as np

def split_known_unknown(data: np.ndarray,
                        labels: np.ndarray,
                        known_classes: list):
    """
    按已知类列表划分数据集：
      train (仅已知类)，
      test_known (同 train)，
      test_unknown (labels ∉ known_classes)
    返回六个数组，方便后续训练与开集评估。 
    """  # np.isin 用于布尔掩码 :contentReference[oaicite:0]{index=0}
    mask = np.isin(labels, known_classes)
    train_data, train_labels = data[mask], labels[mask]
    test_known_data, test_known_labels = train_data, train_labels
    test_unknown_data = data[~mask]; test_unknown_labels = labels[~mask]
    return (train_data, train_labels,
            test_known_data, test_known_labels,
            test_unknown_data, test_unknown_labels)


# # utils/split_known_unknown.py
# import numpy as np
#
# def split_known_unknown(data: np.ndarray,
#                         labels: np.ndarray,
#                         known_classes: list):
#     """
#     将 (data, labels) 按 known_classes 划为：
#       train_data, train_labels,
#       test_known_data, test_known_labels,
#       test_unknown_data, test_unknown_labels
#     """
#     mask = np.isin(labels, known_classes)             # 元素级 in 操作 :contentReference[oaicite:0]{index=0}
#     train_data, train_labels = data[mask], labels[mask]
#     test_known_data, test_known_labels = train_data, train_labels
#     test_unknown_data = data[~mask]; test_unknown_labels = labels[~mask]
#     return (train_data, train_labels,
#             test_known_data, test_known_labels,
#             test_unknown_data, test_unknown_labels)
