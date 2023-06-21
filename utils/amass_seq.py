from torch.utils.data import Dataset
import numpy as np
from utils.load_amass import load_amass
import os

import pickle

import numpy

import torch
from scipy.spatial.transform import Rotation as R

# [ori_head, ori_root, ori_left_arm, ori_left_hand, ori_left_leg, ori_left_ankle,
#                  ori_right_arm, ori_right_hand, ori_right_leg, ori_right_ankle]
# 0:头,1:根,2:左手肘,3:左手腕,4:左膝盖,5:左脚踝,6:右手肘,7:右手腕,8:右膝盖,9:右脚踝
# 选择 根节点，头，左手腕，右手腕，左膝盖，右膝盖
# DIP选择的就是wrist
# IMU_idx = [1, 0, 2, 6, 4, 8]  # 将头与根节点换位置
# IMU_idx = [2, 6, 4, 8, 0, 1]
# 根节点，头，左手肘，左手腕，左膝盖，左脚踝，右手肘，右手腕，右膝盖，右脚踝
# Smpl的关节位置与目前是个关节位置的顺序对应
acc_scale = 30
vel_scale = 3


class amass_rnn(Dataset):
    def __init__(self, path_to_data, input_n, output_n, split=0):
        amass_dir = path_to_data
        tra_val_test = ['train', 'val', 'test', 'pose_test']
        subset_split = tra_val_test[split]
        self.data_path = os.path.join(amass_dir, subset_split)
        self.seq_len = input_n + output_n  # 采样的长度
        self.input_n = input_n
        self.output_n = output_n
        #
        # 数据的长度
        self.data_dir = os.listdir(self.data_path)
        self.lenth = len(self.data_dir)

    # 得到长度
    def __len__(self):
        return self.lenth

    # 得到input ，target，all
    def __getitem__(self, item):
        seq = os.path.join(self.data_path, self.data_dir[item])
        f = open(seq, 'rb')
        data = pickle.load(f, encoding='latin1')
        # 32,50,24,3和32,50,10,3
        all_n = self.input_n + self.output_n
        acc = numpy.array(data['acc'])[:all_n, ]
        ori = numpy.array(data['ori'])[:all_n, ]
        poses = numpy.array(data['poses'])[:all_n, ]
        joints = numpy.array(data['joints'])[:all_n, ]
        # 如果ori的长度不是4，那就已经转化为rotation_matrix格式了。
        if len(ori.shape) != 4:
            ori = axis_angle_to_rotation_matrix(ori)
            # 这里加噪声

        poses = axis_angle_to_rotation_matrix(poses).reshape(self.seq_len, 24, 3, 3)
        # 数据预处理
        acc, ori = normalize_and_concat(acc, glb_ori=ori)
        amass_joints = joints
        amass_poses = poses
        self.input_acc = acc
        self.input_ori = ori
        self.out_poses = amass_poses
        self.out_joints = amass_joints
        return self.input_ori, self.input_acc, self.out_poses, self.out_joints


def normalize_and_concat(glb_acc, glb_ori):
    glb_acc = torch.from_numpy(glb_acc.reshape(-1, 6, 3)).float()
    glb_ori = torch.from_numpy(glb_ori.reshape(-1, 6, 3, 3)).float()
    acc = torch.cat((glb_acc[:, :5] - glb_acc[:, 5:], glb_acc[:, 5:]), dim=1).bmm(glb_ori[:, -1]) / acc_scale
    ori = torch.cat((glb_ori[:, 5:].transpose(2, 3).matmul(glb_ori[:, :5]), glb_ori[:, 5:]), dim=1)
    # 改成根节点在前的
    return acc.numpy(), ori.numpy()


def normalize_tensor(x: torch.Tensor, dim=-1, return_norm=False):
    r"""
    Normalize a tensor in a specific dimension to unit norm. (torch)

    :param x: Tensor in any shape.
    :param dim: The dimension to be normalized.
    :param return_norm: If True, norm(length) tensor will also be returned.
    :return: Tensor in the same shape. If return_norm is True, norm tensor in shape [*, 1, *] (1 at dim)
             will also be returned (keepdim=True).
    """
    norm = x.norm(dim=dim, keepdim=True)
    normalized_x = x / norm
    return normalized_x if not return_norm else (normalized_x, norm)


def vector_cross_matrix(x: torch.Tensor):
    r"""
    Get the skew-symmetric matrix :Math_utils:`[v]_\times\in so(3)` for each vector3 `v`. (torch, batch)

    :param x: Tensor that can reshape to [batch_size, 3].
    :return: The skew-symmetric matrix in shape [batch_size, 3, 3].
    """
    x = x.view(-1, 3)
    zeros = torch.zeros(x.shape[0], device=x.device)
    return torch.stack((zeros, -x[:, 2], x[:, 1],
                        x[:, 2], zeros, -x[:, 0],
                        -x[:, 1], x[:, 0], zeros), dim=1).view(-1, 3, 3)


def axis_angle_to_rotation_matrix(a: torch.Tensor):
    r"""
    Turn axis-angles into rotation matrices. (torch, batch)
    :param a: Axis-angle tensor that can reshape to [batch_size, 3].
    :return: Rotation matrix of shape [batch_size, 3, 3].
    """
    a = torch.from_numpy(a)
    axis, angle = normalize_tensor(a.view(-1, 3), return_norm=True)
    axis[torch.isnan(axis)] = 0  # 如果有nan用0填充
    i_cube = torch.eye(3, device=a.device).expand(angle.shape[0], 3, 3)
    # i_cube(n,3,3), n个I3矩阵
    c, s = angle.cos().view(-1, 1, 1), angle.sin().view(-1, 1, 1)
    r = c * i_cube + (1 - c) * torch.bmm(axis.view(-1, 3, 1), axis.view(-1, 1, 3)) + s * vector_cross_matrix(axis)
    return r.numpy()
