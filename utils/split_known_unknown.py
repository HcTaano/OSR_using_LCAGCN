# utils/split_known_unknown.py
import numpy as np

def split_known_unknown(data: np.ndarray,
                        labels: np.ndarray,
                        known_classes: list):
    """
    将 (data, labels) 按 known_classes 划为：
      train_data, train_labels,
      test_known_data, test_known_labels,
      test_unknown_data, test_unknown_labels
    """
    mask = np.isin(labels, known_classes)             # 元素级 in 操作 :contentReference[oaicite:0]{index=0}
    train_data, train_labels = data[mask], labels[mask]
    test_known_data, test_known_labels = train_data, train_labels
    test_unknown_data = data[~mask]; test_unknown_labels = labels[~mask]
    return (train_data, train_labels,
            test_known_data, test_known_labels,
            test_unknown_data, test_unknown_labels)
