# File: data_utils.py
import os, numpy as np

def load_msr(path):
    """
    加载 MSRAction3D 骨架数据。
    假设路径下为 aXX_sYY_eZZ_skeleton3D.txt 文件，每文件对应一个动作序列。
    返回 (data, labels)，其中 data=[N,T,20,3]，label=[N] (0-based)
    """
    files = sorted(os.listdir(path))
    data_list, label_list = [], []
    for fname in files:
        if not fname.endswith('.txt'): continue
        # 文件名格式 a{action:2d}_s{subject}_e{rep}_skeleton3D.txt
        action = int(fname[1:3]) - 1  # 转为0基
        arr = np.loadtxt(os.path.join(path, fname))
        # 删除置信度列 (若存在)，reshape为帧*关节*3
        if arr.shape[1] == 4:
            arr = arr[:, :3]  # 去掉最后一列
        num_joints = 20
        num_frames = arr.shape[0] // num_joints
        arr = arr.reshape(num_frames, num_joints, 3)
        data_list.append(arr)
        label_list.append(action)
    # 对齐帧长 T=20（超长截断，过短补零）
    T = 20
    N = len(data_list)
    data = np.zeros((N, T, num_joints, 3), dtype=np.float32)
    labels = np.array(label_list, dtype=np.int64)
    for i, arr in enumerate(data_list):
        if arr.shape[0] >= T:
            data[i] = arr[:T]
        else:
            data[i, :arr.shape[0]] = arr  # 补零
    # 转换通道顺序为 [N,3,T,20]
    data = data.transpose(0,3,1,2)
    return data, labels

# def load_utk(path_joints, path_label):
#     """
#     加载 UTKinect 骨架数据。joints.zip 解压后目录包含名为 sXX_eYY_frame... 的文件，
#     actionLabel.txt 文件标注了每个视频中各动作的帧范围。
#     返回 (data, labels)，对每个动作段为一个样本。格式同上 [N,3,T,20]。
#     """
#     # 读取标签文件：lines 格式 "s01_e01 walk:  252 390  sitDown: 572 686  ..."
#     labels_txt = open(path_label).read().split()
#     # 提取动作及帧范围
#     label_info = {}
#     # 按行处理
#     for line in open(path_label):
#         parts = line.strip().split()
#         if not parts: continue
#         vid = parts[0]  # 如 s01_e01
#         label_info[vid] = []
#         idx = 1
#         while idx < len(parts):
#             action = parts[idx].strip(':')
#             start = int(parts[idx+1])
#             end = int(parts[idx+2])
#             label_info[vid].append((action, start, end))
#             idx += 3
#     # 关节文件目录
#     files = sorted(os.listdir(path_joints))
#     data_segments, label_segments = [], []
#     # 定义动作类别映射
#     action_map = {'walk':0,'sitDown':1,'standUp':2,'pickUp':3,'carry':4,
#                   'throw':5,'push':6,'pull':7,'waveHands':8,'clapHands':9}
#     T=20; num_joints=20
#     for fname in files:
#         if not fname.endswith('.txt'): continue
#         vid = fname[:6]  # e.g. s01_e01
#         if vid not in label_info: continue
#         arr = np.loadtxt(os.path.join(path_joints, fname))
#         arr = arr.reshape(-1, num_joints, 4)
#         # 取前三列(x,y,z)
#         coords = arr[:, :, :3]
#         # 对每个标注动作段切割
#         for (action, start, end) in label_info[vid]:
#             seg = coords[start:end]  # shape [frames, 20, 3]
#             num_frames = seg.shape[0]
#             # 对齐到 T 帧
#             seg_data = np.zeros((T, num_joints, 3), dtype=np.float32)
#             if num_frames >= T:
#                 seg_data[:] = seg[:T]
#             else:
#                 seg_data[:num_frames] = seg
#             data_segments.append(seg_data.transpose(2,0,1))  # (3,T,20)
#             label_segments.append(action_map[action])
#     data = np.stack(data_segments, axis=0)  # [N,3,T,20]
#     labels = np.array(label_segments, dtype=np.int64)
#     return data, labels


def load_utk(path_joints, path_label):
    """
    加载 UTKinect 骨架数据，正确解析每行 1 + 20*3 列格式。
    返回 (data, labels)，data 形状 [N,3,T,20]，labels 形状 [N,].
    """
    # --- 1. 读取动作段标签 ---
    label_info = {}
    for line in open(path_label):
        parts = line.strip().split()
        if not parts: continue
        vid = parts[0]  # e.g. 's01_e01'
        label_info[vid] = []
        idx = 1
        while idx + 2 < len(parts):
            action = parts[idx].strip(':')
            start, end = int(parts[idx+1]), int(parts[idx+2])
            label_info[vid].append((action, start, end))
            idx += 3

    # --- 2. 逐文件解析 ---
    files = sorted(os.listdir(path_joints))
    data_list, label_list = [], []
    action_map = {'walk':0,'sitDown':1,'standUp':2,'pickUp':3,'carry':4,
                  'throw':5,'push':6,'pull':7,'waveHands':8,'clapHands':9}
    T, V = 20, 20

    for fname in files:
        if not fname.endswith('.txt'): continue
        vid = fname[:6]
        if vid not in label_info: continue

        # 2.1 加载整表：shape = (num_frames, 61)
        mat = np.loadtxt(os.path.join(path_joints, fname), dtype=np.float32)
        # 2.2 提取坐标部分（跳过第0列帧号），重塑为 (num_frames,20,3)
        coords = mat[:, 1:].reshape(-1, V, 3)

        # 2.3 按段切割并对齐到 T 帧
        for action, s, e in label_info[vid]:
            seg = coords[s:e]  # shape (seg_len,20,3)
            seg_len = seg.shape[0]
            buf = np.zeros((T, V, 3), dtype=np.float32)
            if seg_len >= T:
                buf[:] = seg[:T]
            else:
                buf[:seg_len] = seg
            # 转为 [3,T,20]
            data_list.append(buf.transpose(2,0,1))
            label_list.append(action_map[action])

    data = np.stack(data_list, axis=0)      # [N,3,T,20]
    labels = np.array(label_list, dtype=np.int64)
    return data, labels