from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
import pickle
from tqdm import tqdm
import glob
import os, sys
from os import path as osp


# from smplpytorch.pytorch.smpl_layer import SMPL_Layer


def load_hps_imu(hps_dir):
    # 大概是没有了，需要自己写，包括trans和pose,很奇怪
    return hps_imu


def load_DIP_IMU_nn(dip_dir, sample_rate, seq_len, split):
    subset = ['imu_own_training.npz', 'imu_own_validation.npz', 'imu_own_test.npz']
    sub_dir = subset[split]
    # 数据初始化
    imu_ori = np.array([])
    imu_acc = np.array([])
    dataset_dir = os.path.join(dip_dir, sub_dir)
    print(dataset_dir)
    data = np.load(dataset_dir, allow_pickle=True)
    print("imu_data", data.files)
    print(type(data))
    data_id = data['data_id']
    orientation = data['orientation']
    acceleration = data['acceleration']
    print("orientation", orientation[0].shape)
    for i in range(len(orientation)):
        if i == 0:
            imu_acc = acceleration[i]
            imu_ori = orientation[i]
        else:
            imu_acc = np.append(imu_acc, acceleration[i], axis=0)
            imu_ori = np.append(imu_ori, orientation[i], axis=0)
    print("acceleration", imu_acc.shape)
    print("imu_ori", imu_ori.shape)
    imu_acc = imu_acc.view(-1, 5, 3)
    imu_ori = imu_ori.view(-1, 5, 9)


def load_DIP(dip_dir):
    subset_dir = [x for x in os.listdir(dip_dir)
                  if os.path.isdir(dip_dir + '/' + x)]
    print("subset_dir", subset_dir)
    gt_smpl_17 = np.array([])
    imu_ori = np.array([])
    imu_acc = np.array([])
    sop_smpl_6 = np.array([])
    sip_smpl_6 = np.array([])
    # 这里是实验的，等正式的时候需要是个for循环
    for subset in subset_dir:
        # 数据输入
        # 数据的拼接
        seqs = glob.glob(os.path.join(dip_dir, subset, '*.pkl'))
        print('-- processing subset {:s}'.format(subset))
        for seq in tqdm(seqs):
            # read data
            f = open(seq, 'rb')
            data = pickle.load(f, encoding='latin1')
            # pprint.pprint(data)
            # 获取data的数据类型
            print("类别", type(data))
            print("imu_acc的数量", data['imu_acc'].shape)
            # 这些都是数组类别的，就可以直接拼接
            if not gt_smpl_17.any():
                print('首先赋值')
                gt_smpl_17 = np.array(data['gt'])
                imu_ori = np.array(data['imu_ori'])
                imu_acc =np.array( data['imu_acc'])
                sop_smpl_6 =np.array( data['sop'])
                sip_smpl_6 = np.array(data['sip'])
            else:
                gt_smpl_17 = np.append(gt_smpl_17, data['gt'], axis=0)
                imu_ori = np.append(imu_ori, data['imu_ori'], axis=0)
                imu_acc = np.append(imu_acc, data['imu_acc'], axis=0)
                sop_smpl_6 = np.append(sop_smpl_6, data['sop'], axis=0)
                sop_smpl_6 = np.append(sip_smpl_6, data['sip'], axis=0)


    dip_imu = {
        'gt_smpl_17': gt_smpl_17,
        'imu_ori': imu_ori,
        'imu_acc': imu_acc,
        'sop_smpl_6': sop_smpl_6,
        'sip_smpl_6': sip_smpl_6
    }
    return dip_imu


def vis_smpl(smpl_data, comp_device):
    # define body model according to gender
    subject_gender = 'female'
    body_model_dir = 'G:/python/Work-Skin-IMU/'
    bm_smplx_fname = osp.join(body_model_dir, 'body_models/smplx/{}/model.npz'.format(subject_gender))

    bm = BodyModel(bm_fname=bm_smplx_fname).to(comp_device)
    faces = c2c(bm.f)  # copytocpu
    # 身体参数
    body_parms = {
        'root_orient': torch.Tensor(smpl_data[:, :3]).to(comp_device),
        # controls the global root orientation 控制根节点的方向
        'pose_body': torch.Tensor(smpl_data[:, 3:66]).to(comp_device),  # controls the body身体运动
        'pose_hand': torch.Tensor(smpl_data[:, 66:]).to(comp_device),
        # controls the finger articulation手指细节
        'trans': torch.Tensor(bdata['trans']).to(comp_device),  # controls the global body position身体位移
        'betas': torch.Tensor(
            np.repeat(bdata['betas'][:num_betas][np.newaxis], repeats=time_length, axis=0)).to(
            comp_device),  # controls the body shape. Body shape is static 身体形状
    }


if __name__ == '__main__':
    # def load_amass_data():
    imu_dataset_path = '/python/dataset/IMU Dataset'
    comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("comp_device", comp_device)
    # dip_nn_dir = os.path.join(imu_dataset_path, 'DIP_IMU_and_Others/DIP_IMU_nn')
    # load_DIP_IMU_nn(dip_dir=dip_nn_dir, sample_rate=2, seq_len=45, split=2)
    # train的数据每个都是(300,45)和(300,15),可以参考来进行一下组合，反正是统一的嘛，直接来个view(n,-1)就可以了
    # 然后test和validation就是train除了那个之外剩下的部分。
    # 载入数据，结合到一起，然后下一步需要整理数据
    dip_dir = os.path.join(imu_dataset_path, 'DIP_IMU_and_Others/DIP_IMU')
    dip_imu = load_DIP(dip_dir)
