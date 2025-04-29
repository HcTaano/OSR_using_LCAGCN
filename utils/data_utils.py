# utils/data_utils.py

import os
import numpy as np

def load_msr(path, T=20, V=20):
    """
    加载 MSRAction3D 骨架数据，返回 data:[N,3,T,V], labels:[N].
    增加随机时间抖动与坐标噪声的数据增强。
    """
    files = sorted(os.listdir(path))
    data_list, label_list = [], []
    for fname in files:
        if not fname.endswith('.txt'): continue
        action = int(fname[1:3]) - 1  # aXX→类别索引0基
        # arr = np.loadtxt(os.path.join(path, fname))
        # 明确以 float32 加载，避免默认 float64 → PyTorch DoubleTensor
        arr = np.loadtxt(os.path.join(path, fname), dtype=np.float32)
        if arr.shape[1] == 4:
            arr = arr[:, :3]          # 丢弃置信度列
        num_frames = arr.shape[0] // V
        arr = arr.reshape(num_frames, V, 3)

        # —— 数据增强 ——
        # 随机时间抖动：偏移起始帧±2，并裁剪或重复边缘帧
        offset = np.random.randint(-2, 3)
        idx = np.clip(np.arange(offset, offset + T), 0, num_frames-1)
        buf = arr[idx]
        # 随机高斯噪声
        buf = buf + np.random.normal(scale=0.01, size=buf.shape).astype(np.float32)

        data_list.append(buf.transpose(2,0,1))  # (3,T,V)
        label_list.append(action)

    data = np.stack(data_list, axis=0)
    labels = np.array(label_list, dtype=np.int64)
    return data, labels

def load_utk(path_joints, path_label, T=20, V=20):
    """
    加载 UTKinect 骨架数据，返回 data:[N,3,T,V], labels:[N].
    保持原有解析逻辑，不再额外处理 NaN。
    """
    # 解析标签
    tokens = open(path_label).read().split()
    label_info = {}
    i = 0
    while i < len(tokens):
        vid = tokens[i]; label_info[vid] = []; i+=1
        while i+2 < len(tokens) and tokens[i].endswith(':'):
            act = tokens[i].rstrip(':')
            s, e = tokens[i+1], tokens[i+2]
            # 仅当 s,e 为数字时才解析
            if s.isdigit() and e.isdigit():
                label_info[vid].append((act, int(s), int(e)))
            i += 3

    # 下钻到 joints 子目录（若存在）
    if os.path.isdir(os.path.join(path_joints, 'joints')):
        path_joints = os.path.join(path_joints, 'joints')

    files = sorted([f for f in os.listdir(path_joints)
                    if f.endswith('.txt') and f.startswith('joints_')])

    data_list, label_list = [], []
    action_map = {'walk':0,'sitDown':1,'standUp':2,'pickUp':3,'carry':4,
                  'throw':5,'push':6,'pull':7,'waveHands':8,'clapHands':9}

    for fname in files:
        base = os.path.splitext(fname)[0].split('_')
        vid = base[1] + '_' + base[2]
        if vid not in label_info: continue
        # mat = np.loadtxt(os.path.join(path_joints, fname), dtype=np.float32)
        # 同样指定 float32，保证 downstream Tensor 为 FloatTensor
        mat = np.loadtxt(os.path.join(path_joints, fname), dtype=np.float32)
        coords = mat[:,1:].reshape(-1, V, 3)

        for act, s, e in label_info[vid]:
            seg = coords[s:e]
            buf = np.zeros((T, V, 3), dtype=np.float32)
            buf[:min(len(seg),T)] = seg[:T]
            data_list.append(buf.transpose(2,0,1))
            label_list.append(action_map[act])

    assert data_list, "UTK 数据加载失败"  # 防止空列表
    # return np.stack(data_list,0), np.array(label_list, dtype=np.int64)
    # 最终确保 data 是 float32
    data = np.stack(data_list,0).astype(np.float32)
    labels = np.array(label_list, dtype=np.int64)
    return data, labels



# # utils/data_utils.py
#
# import os
# import numpy as np
#
# def load_msr(path, T=20, V=20):
#     """
#     加载 MSRAction3D 骨架数据（Real3D 坐标）。
#     每个文件 aXX_sYY_eZZ_skeleton3D.txt 包含若干帧、
#     每帧 20 个关节的 (x,y,z) 坐标。
#     返回：
#       data: numpy.ndarray, 形状 [N,3,T,V]
#       labels: numpy.ndarray, 形状 [N,]
#     """  # ST-GCN 通用输入格式[N,C,T,V]
#     files = sorted(os.listdir(path))
#     data_list, label_list = [], []
#     for fname in files:
#         if not fname.endswith('.txt'):
#             continue
#         # 从文件名提取动作编号：a01→0 基
#         action = int(fname[1:3]) - 1
#         arr = np.loadtxt(os.path.join(path, fname))  # 读入所有行
#         # 若为 (u,v,d,c) 格式，则丢弃置信度列
#         if arr.shape[1] == 4:
#             arr = arr[:, :3]
#         num_frames = arr.shape[0] // V
#         arr = arr.reshape(num_frames, V, 3)  # 重塑为 (帧数, 关节数, 3)
#         # 对齐到固定帧长 T：截断或零填充
#         buf = np.zeros((T, V, 3), dtype=np.float32)
#         buf[:min(num_frames, T)] = arr[:T]
#         # 转置为 (3,T,V) 以符合 PyTorch 定常 [C,T,V]
#         data_list.append(buf.transpose(2,0,1))
#         label_list.append(action)
#     # 堆叠所有样本，形成 [N,3,T,V]
#     data = np.stack(data_list, axis=0)
#     labels = np.array(label_list, dtype=np.int64)
#     return data, labels
#
# def load_utk(path_joints, path_label, T=20, V=20):
#     """
#        加载 UTKinect-Action3D 骨架数据（每行 1 + 20*3 列）。
#        返回 data: [N,3,T,20], labels: [N].
#     """
#     # —— 1. 一次性读取全部 tokens
#     tokens = open(path_label).read().split()
#     label_info = {}
#     i = 0
#     # —— 2. 按 vid + 若干 (action, start, end) 解析
#     while i < len(tokens):
#         vid = tokens[i]                # e.g. 's01_e01'
#         label_info[vid] = []
#         i += 1
#         # 若下一个 token 以 ':' 结尾，说明是动作名
#         while i+2 < len(tokens) and tokens[i].endswith(':'):
#             act = tokens[i].rstrip(':')
#             s, e = int(tokens[i+1]), int(tokens[i+2])
#             label_info[vid].append((act, s, e))
#             i += 3
#     # 之后的逻辑同前：下钻子目录、过滤文件、reshape、对齐…
# # def load_utk(path_joints, path_label, T=20, V=20):
# #     # 1. 读取动作段标签
# #     label_info = {}
# #     for line in open(path_label):
# #         parts = line.strip().split()
# #         if not parts: continue
# #         vid = parts[0]
# #         label_info[vid] = []
# #         idx = 1
# #         while idx + 2 < len(parts):
# #             act = parts[idx].strip(':')
# #             s, e = int(parts[idx+1]), int(parts[idx+2])
# #             label_info[vid].append((act, s, e))
# #             idx += 3
#
#     # 自动下钻至真正存放骨架 .txt 的子目录
#     if os.path.isdir(os.path.join(path_joints, 'joints')):
#         path_joints = os.path.join(path_joints, 'joints')  # 递归下钻
#
#     # 仅保留以 joints_ 开头的骨架文件，过滤掉 actionLabel.txt 等
#     files = sorted([
#         f for f in os.listdir(path_joints)
#         if f.endswith('.txt') and f.startswith('joints_')
#     ])  # 这样 files 中就只剩 ['joints_s01_e01.txt', …]
#
#     data_list, label_list = [], []
#     action_map = {'walk':0,'sitDown':1,'standUp':2,'pickUp':3,'carry':4,
#                   'throw':5,'push':6,'pull':7,'waveHands':8,'clapHands':9}
#
#     # 调试打印
#     print("UTK label videos:", list(label_info.keys())[:5])
#     print("UTK skeleton files:", files[:5])
#
#     # —— 4. 逐文件加载、切片、对齐
#     for fname in files:
#         # if not fname.endswith('.txt'): continue
#         # 通用提取 'sXX_eYY'
#         # 只处理 joints_sXX_eYY*.txt
#         # vid 格式应为 'sXX_eYY'
#         base = os.path.splitext(fname)[0]
#         # 举例 'joints_s01_e01_frame001' → ['joints','s01','e01','frame001']
#         parts = base.split('_')
#         vid = parts[1] + '_' + parts[2]          # 's01'+'_'+'e01'
#         if vid not in label_info:
#             continue
#
#         mat = np.loadtxt(os.path.join(path_joints, fname), dtype=np.float32)
#         coords = mat[:,1:].reshape(-1, V, 3)
#
#         for act, s, e in label_info[vid]:
#             seg = coords[s:e]
#             buf = np.zeros((T, V, 3), dtype=np.float32)
#             buf[:min(len(seg),T)] = seg[:T]
#             data_list.append(buf.transpose(2,0,1))
#             label_list.append(action_map[act])
#
#     # —— 5. 检查并返回
#     # 断言确保非空
#     assert data_list, "ERROR: 没有读取到任何段，请检查文件名与标签 vid 是否对得上"
#     data = np.stack(data_list, axis=0)
#     labels = np.array(label_list, dtype=np.int64)
#     return data, labels
#
# # def load_utk(path_joints, path_label, T=20, V=20):
# #     """
# #     加载 UTKinect-Action3D 骨架数据。
# #     每行首列是帧序号，后续 20*3 列为 20 个关节的 (x,y,z)。
# #     actionLabel.txt 指定每个视频的动作段区间。
# #     返回同样的 [N,3,T,V] 和 [N,] 标签。
# #     """  # UTKinect 格式说明
# #     # 解析动作段标签
# #     label_info = {}
# #     for line in open(path_label):
# #         parts = line.strip().split()
# #         if not parts:
# #             continue
# #         vid = parts[0]  # 如 's01_e01'
# #         label_info[vid] = []
# #         idx = 1
# #         while idx + 2 < len(parts):
# #             act = parts[idx].strip(':')
# #             s, e = int(parts[idx+1]), int(parts[idx+2])
# #             label_info[vid].append((act, s, e))
# #             idx += 3
# #
# #     action_map = {'walk':0,'sitDown':1,'standUp':2,'pickUp':3,'carry':4,
# #                   'throw':5,'push':6,'pull':7,'waveHands':8,'clapHands':9}
# #
# #     files = sorted(os.listdir(path_joints))
# #     data_list, label_list = [], []
# #     for fname in files:
# #         if not fname.endswith('.txt'):
# #             continue
# #         vid = fname[:6]
# #         if vid not in label_info:
# #             continue
# #
# #         mat = np.loadtxt(os.path.join(path_joints, fname), dtype=np.float32)
# #         # 丢弃第 0 列（帧号），重塑剩余 60 列→(帧数,20,3)
# #         coords = mat[:,1:].reshape(-1, V, 3)
# #
# #         # 按每个标注段切割并对齐到 T 帧
# #         for act, s, e in label_info[vid]:
# #             seg = coords[s:e]
# #             buf = np.zeros((T, V, 3), dtype=np.float32)
# #             buf[:min(len(seg),T)] = seg[:T]
# #             data_list.append(buf.transpose(2,0,1))
# #             label_list.append(action_map[act])
# #     data = np.stack(data_list, axis=0)
# #     labels = np.array(label_list, dtype=np.int64)
# #     return data, labels
#
#
# # # File: data_utils.py
# # import os, numpy as np
# #
# # def load_msr(path):
# #     """
# #     加载 MSRAction3D 骨架数据。
# #     假设路径下为 aXX_sYY_eZZ_skeleton3D.txt 文件，每文件对应一个动作序列。
# #     返回 (data, labels)，其中 data=[N,T,20,3]，label=[N] (0-based)
# #     """
# #     files = sorted(os.listdir(path))
# #     data_list, label_list = [], []
# #     for fname in files:
# #         if not fname.endswith('.txt'): continue
# #         # 文件名格式 a{action:2d}_s{subject}_e{rep}_skeleton3D.txt
# #         action = int(fname[1:3]) - 1  # 转为0基
# #         arr = np.loadtxt(os.path.join(path, fname))
# #         # 删除置信度列 (若存在)，reshape为帧*关节*3
# #         if arr.shape[1] == 4:
# #             arr = arr[:, :3]  # 去掉最后一列
# #         num_joints = 20
# #         num_frames = arr.shape[0] // num_joints
# #         arr = arr.reshape(num_frames, num_joints, 3)
# #         data_list.append(arr)
# #         label_list.append(action)
# #     # 对齐帧长 T=20（超长截断，过短补零）
# #     T = 20
# #     N = len(data_list)
# #     data = np.zeros((N, T, num_joints, 3), dtype=np.float32)
# #     labels = np.array(label_list, dtype=np.int64)
# #     for i, arr in enumerate(data_list):
# #         if arr.shape[0] >= T:
# #             data[i] = arr[:T]
# #         else:
# #             data[i, :arr.shape[0]] = arr  # 补零
# #     # 转换通道顺序为 [N,3,T,20]
# #     data = data.transpose(0,3,1,2)
# #     return data, labels
# #
# # # def load_utk(path_joints, path_label):
# # #     """
# # #     加载 UTKinect 骨架数据。joints.zip 解压后目录包含名为 sXX_eYY_frame... 的文件，
# # #     actionLabel.txt 文件标注了每个视频中各动作的帧范围。
# # #     返回 (data, labels)，对每个动作段为一个样本。格式同上 [N,3,T,20]。
# # #     """
# # #     # 读取标签文件：lines 格式 "s01_e01 walk:  252 390  sitDown: 572 686  ..."
# # #     labels_txt = open(path_label).read().split()
# # #     # 提取动作及帧范围
# # #     label_info = {}
# # #     # 按行处理
# # #     for line in open(path_label):
# # #         parts = line.strip().split()
# # #         if not parts: continue
# # #         vid = parts[0]  # 如 s01_e01
# # #         label_info[vid] = []
# # #         idx = 1
# # #         while idx < len(parts):
# # #             action = parts[idx].strip(':')
# # #             start = int(parts[idx+1])
# # #             end = int(parts[idx+2])
# # #             label_info[vid].append((action, start, end))
# # #             idx += 3
# # #     # 关节文件目录
# # #     files = sorted(os.listdir(path_joints))
# # #     data_segments, label_segments = [], []
# # #     # 定义动作类别映射
# # #     action_map = {'walk':0,'sitDown':1,'standUp':2,'pickUp':3,'carry':4,
# # #                   'throw':5,'push':6,'pull':7,'waveHands':8,'clapHands':9}
# # #     T=20; num_joints=20
# # #     for fname in files:
# # #         if not fname.endswith('.txt'): continue
# # #         vid = fname[:6]  # e.g. s01_e01
# # #         if vid not in label_info: continue
# # #         arr = np.loadtxt(os.path.join(path_joints, fname))
# # #         arr = arr.reshape(-1, num_joints, 4)
# # #         # 取前三列(x,y,z)
# # #         coords = arr[:, :, :3]
# # #         # 对每个标注动作段切割
# # #         for (action, start, end) in label_info[vid]:
# # #             seg = coords[start:end]  # shape [frames, 20, 3]
# # #             num_frames = seg.shape[0]
# # #             # 对齐到 T 帧
# # #             seg_data = np.zeros((T, num_joints, 3), dtype=np.float32)
# # #             if num_frames >= T:
# # #                 seg_data[:] = seg[:T]
# # #             else:
# # #                 seg_data[:num_frames] = seg
# # #             data_segments.append(seg_data.transpose(2,0,1))  # (3,T,20)
# # #             label_segments.append(action_map[action])
# # #     data = np.stack(data_segments, axis=0)  # [N,3,T,20]
# # #     labels = np.array(label_segments, dtype=np.int64)
# # #     return data, labels
# #
# #
# # def load_utk(path_joints, path_label):
# #     """
# #     加载 UTKinect 骨架数据，正确解析每行 1 + 20*3 列格式。
# #     返回 (data, labels)，data 形状 [N,3,T,20]，labels 形状 [N,].
# #     """
# #     # --- 1. 读取动作段标签 ---
# #     label_info = {}
# #     for line in open(path_label):
# #         parts = line.strip().split()
# #         if not parts: continue
# #         vid = parts[0]  # e.g. 's01_e01'
# #         label_info[vid] = []
# #         idx = 1
# #         while idx + 2 < len(parts):
# #             action = parts[idx].strip(':')
# #             start, end = int(parts[idx+1]), int(parts[idx+2])
# #             label_info[vid].append((action, start, end))
# #             idx += 3
# #
# #     # --- 2. 逐文件解析 ---
# #     files = sorted(os.listdir(path_joints))
# #     data_list, label_list = [], []
# #     action_map = {'walk':0,'sitDown':1,'standUp':2,'pickUp':3,'carry':4,
# #                   'throw':5,'push':6,'pull':7,'waveHands':8,'clapHands':9}
# #     T, V = 20, 20
# #
# #     for fname in files:
# #         if not fname.endswith('.txt'): continue
# #         vid = fname[:6]
# #         if vid not in label_info: continue
# #
# #         # 2.1 加载整表：shape = (num_frames, 61)
# #         mat = np.loadtxt(os.path.join(path_joints, fname), dtype=np.float32)
# #         # 2.2 提取坐标部分（跳过第0列帧号），重塑为 (num_frames,20,3)
# #         coords = mat[:, 1:].reshape(-1, V, 3)
# #
# #         # 2.3 按段切割并对齐到 T 帧
# #         for action, s, e in label_info[vid]:
# #             seg = coords[s:e]  # shape (seg_len,20,3)
# #             seg_len = seg.shape[0]
# #             buf = np.zeros((T, V, 3), dtype=np.float32)
# #             if seg_len >= T:
# #                 buf[:] = seg[:T]
# #             else:
# #                 buf[:seg_len] = seg
# #             # 转为 [3,T,20]
# #             data_list.append(buf.transpose(2,0,1))
# #             label_list.append(action_map[action])
# #
# #     data = np.stack(data_list, axis=0)      # [N,3,T,20]
# #     labels = np.array(label_list, dtype=np.int64)
# #     return data, labels