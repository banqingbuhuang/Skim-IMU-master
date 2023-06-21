from torch.utils.data import Dataset
import numpy as np
from utils.load_amass import load_amass
import os

import pickle
import smplx
import numpy

import torch
from scipy.spatial.transform import Rotation as R
'''
单个样本，并不是窗口移动的采样策略，用于可视化和测试
'''
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
imu_mask = [7, 8, 11, 12, 0, 2]
TARGET_FPS=60


class amass_rnn(Dataset):
    def __init__(self, path_to_data,input_n, output_n, split=0):
        amass_dir = path_to_data
        self.seq_len = input_n + output_n  # 采样的长度
        self.input_n = input_n
        self.output_n = output_n
        #
        # 数据的长度
        ori, acc, poses, joints = load_data(path_to_data)
        # 32,50,24,3和32,50,10,3
        # 数据预处
        amass_acc = acc
        amass_ori = ori
        amass_joints = joints
        amass_poses = poses
        self.input_acc = amass_acc
        self.input_ori = amass_ori
        self.out_poses = amass_poses
        self.out_joints = amass_joints
        self.lenth = len(self.out_joints)
        print(self.input_ori.shape, self.input_acc.shape, self.out_joints.shape, self.out_poses.shape)

    # 得到长度
    def __len__(self):
        return np.shape(self.input_ori)[0]

    # 得到input ，target，all
    def __getitem__(self, item):
        return self.input_ori[item], self.input_acc[item], self.out_poses[item], self.out_joints[item]


def normalize_and_concat(glb_acc, glb_ori):
    glb_acc = glb_acc.reshape(-1, 6, 3).float()
    glb_ori = glb_ori.reshape(-1, 6, 3, 3).float()
    acc = torch.cat((glb_acc[:, :5] - glb_acc[:, 5:], glb_acc[:, 5:]), dim=1).bmm(glb_ori[:, -1]) / acc_scale
    ori = torch.cat((glb_ori[:, 5:].transpose(2, 3).matmul(glb_ori[:, :5]), glb_ori[:, 5:]), dim=1)
    # 改成根节点在前的
    return acc, ori


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


def load_data(path):
    amass_acc_sam, amass_ori_sam, amass_poses_sam, amass_joints_sam = np.array([]), np.array([]), np.array(
        []), np.array([])
    data = pickle.load(open(path, 'rb'), encoding='latin1')
    ori = torch.from_numpy(data['ori']).float()
    acc = torch.from_numpy(data['acc']).float()
    poses = torch.from_numpy(data['gt']).float()
    # fill nan with nearest neighbors
    min_num = min(acc.shape[0], poses.shape[0])
    acc = acc[:min_num]
    ori = ori[:min_num]
    poses = poses[:min_num]
    print("shape", ori.shape, acc.shape, poses.shape)
    acc, ori = normalize_and_concat(acc, glb_ori=ori)

    if torch.isnan(acc).sum() == 0 and torch.isnan(ori).sum() == 0 and torch.isnan(poses).sum() == 0:
        mocap_framerate = 60
        if mocap_framerate % TARGET_FPS == 0:
            n_tmp = int(mocap_framerate / TARGET_FPS)
            acc = acc[::n_tmp].numpy()
            ori = ori[::n_tmp].numpy()
            poses = poses[::n_tmp].numpy()
        n_frames = poses.shape[0]
        # smpl_folder = "/data/wwu/xt/body_models/vPOSE/models"
        smpl_folder = "C:/Gtrans/body_models/vPOSE/models"
        model_type = 'smpl'
        ext = 'pkl'
        num_expression_coeffs = 10
        use_face_contour = False

        gender = 'male'
        num_betas = 10
        betas = np.zeros(num_betas)
        root_orient = torch.Tensor(poses[:, :3])
        body_pose = torch.Tensor(poses[:, 3:72])
        betas = torch.Tensor(np.repeat(betas[:num_betas][np.newaxis], repeats=n_frames, axis=0))

        model = smplx.create(model_path=smpl_folder, model_type=model_type,
                             gender=gender, num_betas=num_betas, use_face_contour=use_face_contour,
                             num_expression_coeffs=num_expression_coeffs,
                             ext=ext)
        expression = torch.randn(
            [1, model.num_expression_coeffs], dtype=torch.float32)
        output = model(global_orient=root_orient, betas=betas,
                       body_pose=body_pose, expression=expression,
                       return_verts=True)
        joints = output.joints.detach().numpy().squeeze()
        joints_pos = joints[:, :24, :]
        print(joints_pos.shape)
        num_frames = len(joints_pos)
        poses = axis_angle_to_rotation_matrix(poses).reshape(num_frames, 24, 3, 3)
        seq_len = 60
        # num_batch = num_frames // seq_len
        # amass_acc_sam = acc[:num_batch * seq_len, :].reshape(num_batch, seq_len, 6, 3)
        # amass_ori_sam = ori[:num_batch * seq_len, :].reshape(num_batch, seq_len, 6, 3, 3)
        # amass_poses_sam = poses[:num_batch * seq_len, :].reshape(num_batch, seq_len, 24, 3, 3)
        # amass_joints_sam = joints_pos[:num_batch * seq_len, :].reshape(num_batch, seq_len, 24, 3)
        fs = np.arange(0, num_frames - seq_len + 1)  # 窗口滑动
        fs_sel = fs
        for i in np.arange(seq_len - 1):
            fs_sel = np.vstack((fs_sel, fs + i + 1))  # 沿着竖直方向将矩阵堆叠起来将fs_sel和fs+i+1放到一起
        fs_sel = fs_sel.transpose()  # eg.((35, 1704)变成(1704, 35))
        amass_acc_sam = acc[fs_sel, :]
        amass_ori_sam = ori[fs_sel, :]
        amass_poses_sam = poses[fs_sel, :]
        amass_joints_sam = joints_pos[fs_sel, :]
    return amass_ori_sam, amass_acc_sam, amass_poses_sam, amass_joints_sam
