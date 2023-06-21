from torch.utils.data import Dataset
import numpy as np
from utils.load_DIP_Model import loadDIP_nn
import torch
import os


class dip_imu(Dataset):
    def __init__(self, path_to_data, input_n=25, output_n=10, sample_rate=2, split=0):
        self.input_n = None
        seq_len = input_n + output_n  # 采样的长度
        tra_val_test = ['train', 'val', 'test']
        subset_split = tra_val_test[split]
        self.data_path = os.path.join(path_to_data, subset_split)
        data = loadDIP_nn(self.data_path)
        # 输入数据，rotation格式。train和validation和test都在一起，需要分割开。
        acc = np.array(data['acc'])
        ori = np.array(data['ori'])
        poses = np.array(data['poses'])
        joints = np.array(data['joints'])
        batch, frame, node, _ = acc.shape

        pad_idx = np.repeat([input_n - 1], output_n)  # 25个(10-1)
        i_idx = np.append(np.arange(0, input_n), pad_idx)  # 前十个是输入，后二十五个是最后一个姿势
        amass_acc = acc.transpose(1, 0, 2, 3)
        input_acc = amass_acc[i_idx, :]
        # ori
        # frame,n,node,9

        amass_ori = ori.reshape(batch, frame, node, -1).transpose(1, 0, 2, 3)
        input_ori = amass_ori[i_idx, :]
        self.input_acc = input_acc.transpose(1, 2, 3, 0)
        self.input_ori = input_ori.transpose(1, 2, 3, 0)
        self.out_poses = poses
        self.out_joints = joints

    def __len__(self):
        return np.shape(self.input_acc)[0]

    def __getitem__(self, item):
        return self.input_acc[item], self.input_ori[item], self.out_poses[item], self.out_joints[item]
