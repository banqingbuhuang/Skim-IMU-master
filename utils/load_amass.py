from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os

import pickle

import numpy as np

import torch
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

"""
对于实验数据的分析
ACCAD:      101 MB
BMLhandball:385 MB
CMU:        2.12 GB太大了，先不放进去
MPI_HDM05:  524 MB
SFU:        58.0 MB 
SSM_synced: 8.75 MB
"""
# ori_tmp = [ori_root,ori_head ori_left_arm, ori_left_hand, ori_left_leg, ori_left_ankle,
#                   ori_right_arm, ori_right_hand, ori_right_leg, ori_right_ankle]
# head root left-hand right-hand left-leg right-leg
IMU_idx = [1, 0, 3, 7, 4, 8]  # 将头与根节点换位置
# Smpl的关节位置与目前是个关节位置的顺序对应
Joint_idx = [0, 15, 21, 20, 5, 4, 19, 18, 8, 7]
acc_scale = 30
vel_scale = 3


def normalize_and_concat(glb_acc, glb_ori):
    glb_acc = torch.from_numpy(glb_acc.reshape(-1, 6, 3))
    glb_ori = torch.from_numpy(glb_ori.reshape(-1, 6, 3, 3))
    acc = torch.cat((glb_acc[:, :5] - glb_acc[:, 5:], glb_acc[:, 5:]), dim=1).bmm(glb_ori[:, -1]) / acc_scale
    ori = torch.cat((glb_ori[:, 5:].transpose(2, 3).matmul(glb_ori[:, :5]), glb_ori[:, 5:]), dim=1)
    return acc.numpy(), ori.numpy()


# sample_rate采样间隔吧
def load_amass(amass_dir, sample_rate, seq_len, split):
    sample_acc = []
    sample_ori = []
    sample_poses = []
    sample_joints = []
    complete_acc, complete_ori, complete_poses, complete_joints = [], [], [], []
    # 根据split分配实验数据，
    tra_val_test = ['train', 'val', 'test']
    subset_split = tra_val_test[split]
    data_dir = os.path.join(amass_dir, subset_split)
    subset_dir = [x for x in os.listdir(data_dir)
                  if os.path.isdir(data_dir + '/' + x)]
    # 整理数据
    # 这里是实验的，等正式的时候需要是个for循环
    for subset in subset_dir:
        # subset = subset_dir[0]
        """
        amass_acc = np.array([])
        amass_ori = np.array([])
        amass_poses = np.array([])
        amass_joints = np.array([])
        """
        # 数据输入
        # 数据的拼接
        seqs = glob.glob(os.path.join(data_dir, subset, '*.pkl'))
        print('-- processing subset {:s}'.format(subset))
        for seq in tqdm(seqs):
            # read data
            f = open(seq, 'rb')
            data = pickle.load(f, encoding='latin1')
            # print("类别", type(data))
            if len(np.array(data['acc']).shape) == 1:
                continue
            # 下采样部分
            n = len(data['acc'])
            even_list = range(0, n, sample_rate)  # [0...n]2为步长去掉一半的动作坐标减少序列长度

            acc = np.array(np.array(data['acc'])[even_list, :])
            ori = np.array(np.array(data['ori'])[even_list, :])
            poses = np.array(np.array(data['poses'])[even_list, :])
            joints = np.array(np.array(data['joints'])[even_list, :])

            # 从10个数据里面提取出六个数据
            # head root left-hand right-hand left-leg right-leg
            acc = np.concatenate([acc[:, idx, :] for idx in IMU_idx], axis=1)
            ori = np.concatenate([ori[:, idx, :] for idx in IMU_idx], axis=1)
            r = R.from_rotvec(ori.reshape(-1, 3))
            ori = np.array(r.as_matrix())
            p = R.from_rotvec(poses.reshape(-1, 3))
            poses = np.array(p.as_matrix()).reshape(-1, 24, 9)

            # 数据预处理
            acc, ori = normalize_and_concat(acc, glb_ori=ori)
            # 变成torch的了

            amass_acc = acc.reshape(-1, len(IMU_idx), 3)
            amass_ori = ori.reshape(-1, len(IMU_idx), 9)
            amass_joints = joints
            amass_poses = poses
            # 滑窗模块
            # 窗口滑动，得到堆叠模块
            num_frames = len(amass_ori)
            fs = np.arange(0, num_frames - seq_len + 1)  # 窗口滑动
            fs_sel = fs
            for i in np.arange(seq_len - 1):
                fs_sel = np.vstack((fs_sel, fs + i + 1))  # 沿着竖直方向将矩阵堆叠起来将fs_sel和fs+i+1放到一起
            fs_sel = fs_sel.transpose()  # eg.((35, 1704)变成(1704, 35))
            amass_acc_sam = amass_acc[fs_sel, :]
            amass_ori_sam = amass_ori[fs_sel, :]
            amass_poses_sam = amass_poses[fs_sel, :]
            amass_joints_sam = amass_joints[fs_sel, :]
            # 将数据拼接到一起
            if len(sample_acc) == 0:
                sample_ori, sample_poses, sample_acc, sample_joints = amass_ori_sam, amass_poses_sam, amass_acc_sam, amass_joints_sam
            else:
                sample_ori = np.concatenate((sample_ori, amass_ori_sam), axis=0)
                sample_poses = np.concatenate((sample_poses, amass_poses_sam), axis=0)
                sample_acc = np.concatenate((sample_acc, amass_acc_sam), axis=0)
                sample_joints = np.concatenate((sample_joints, amass_joints_sam), axis=0)
                """
                complete_acc = np.append(complete_acc, amass_acc, axis=0)
                complete_ori = np.append(complete_ori, amass_ori, axis=0)
                complete_poses = np.append(complete_poses, amass_poses, axis=0)
                complete_joints = np.append(complete_joints, amass_joints, axis=0)
                """
    amass_imu = {
        'ori': sample_ori,
        'acc': sample_acc,
        'poses': sample_poses,
        'joints': sample_joints
    }
    # (68511, 45, 6, 3)
    # (68511, 45, 6, 3, 3)
    # (68511, 45, 135)

    return amass_imu


if __name__ == "__main__":
    amass_dir = '/data/xt/dataset/MPI_HDM05'
    # amass_imu = load_amass(amass_dir, sample_rate=2, seq_len=50, split=0)
    # print("zuizhong", amass_imu['joints'].shape)
    path = "C:/Gtrans/dataset/AMASS/AMASSCA/all_data/ACCADFemale1General_c3d50.pkl"
    f = open(path, 'rb')
    data = pickle.load(f, encoding='latin1')
    print("data", data['acc'].shape)
    print("data", data['ori'].shape)
    print("data", data['joints'].shape)
    print("data", data['poses'].shape)
