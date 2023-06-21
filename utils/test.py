import utils.viz as viz

import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
import os
from matplotlib import pyplot as plt
import numpy as np
from utils.single_total import amass_rnn as total
from utils.single_dip import amass_rnn as dip
import model.RNN_best_total as nnmodel_1
import model.GCN as nnmodel_2
import model.Seq2Seq as residual
import model.RNN_6_pose_DIP as nn_pose
from utils.opt import Options
from utils import loss_func
from utils.amass_6_node import amass_rnn as amass

node_ignore = [0, 1, 2, 3, 6, 9, 12, 13, 14, 16, 17]
node_ignore = torch.as_tensor(node_ignore)


def eval(opt):
    input_n = opt.input_n
    output_n = opt.output_n
    all_n = input_n + output_n
    data_path = "/data/wwu/xt/dataset/TotalCapture_Real/test/acting/s3_acting1.pkl"
    print(" torch.cuda.is_available()", torch.cuda.is_available())
    # device = torch.device("cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    adjs = opt.adjs
    is_cuda = False
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    script_name = os.path.basename(__file__).split('.')[0]

    # new_3:将up变成LSTM
    script_name += "eval_rnn_pro_2_in{:d}_out{:d}_dctn{:d}".format(opt.input_n, opt.output_n, opt.all_n)
    # 返回main_in10_out25_dctn35
    print(">>> creating model")

    model_pro = nnmodel_1.Predict_imu(input_frame_len=all_n, output_frame_len=all_n, input_size=72,
                                      mid_size=18, output_size=72, adjs=adjs,
                                      device=device, dropout=0.2)
    model_pro = model_pro.to(device)
    model_pro_path = "/data/wwu/xt/IMU/checkpoint/test/ckpt_main_rnn_gcn_pron222_in36_out24_dctn60_best.pth.tar"
    print(">>> loading ckpt len from '{}'".format(model_pro_path))
    if is_cuda:
        ckpt = torch.load(model_pro_path)
    else:
        ckpt = torch.load(model_pro_path, map_location='cpu')
    start_epoch = ckpt['epoch']
    print(">>>  start_epoch", start_epoch)
    model_pro.load_state_dict(ckpt['state_dict'])
    print(">>> ckpt len loaded (epoch: {} )".format(start_epoch))
    test_dataset = total(path_to_data=data_path, input_n=input_n, output_n=output_n,
                         split=2)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=256,  # 128
        shuffle=False,
        num_workers=0,  # 原来写的是opt.job=10，现在因为虚拟内存不够，更改为0。
        pin_memory=True)
    model_pro.eval()
    N = 0
    eval_frame = [5, 11, 17, 23, 29, 35, 36, 38, 41, 44, 47, 50, 53, 56, 59]
    t_posi = np.zeros(len(eval_frame))
    for i, (input_ori, input_acc, out_poses, out_joints) in enumerate(test_loader):
        if torch.isnan(input_ori).sum() == 0 and torch.isnan(input_acc).sum() == 0 \
                and torch.isnan(out_poses).sum() == 0 and torch.isnan(out_joints).sum() == 0:
            if torch.cuda.is_available():
                # model_input = model_input.to(device).float()
                input_ori = input_ori.to(device).float()
                input_acc = input_acc.to(device).float()
                out_joints = out_joints.to(device).float()
            # model要改
            y_out = model_pro(input_ori, input_acc)
            batch, frame, _, _ = y_out.data.shape
            for k in np.arange(0, len(eval_frame)):
                j = eval_frame[k]
                test_out, test_joints = y_out[:, j, :], out_joints[:, j, :]
                t_posi[k] += loss_func.position_loss(test_out, test_joints).cpu().data.numpy() * batch * 100

            N += batch
        print(t_posi / N)


def demo_benji(opt):
    input_n = opt.input_n
    output_n = opt.output_n
    all_n = input_n + output_n
    data_path = "C:/Gtrans/dataset/IMU Dataset/DIP_IMU/test/s_10/01_a.pkl"
    adjs = opt.adjs
    is_cuda = False
    device = torch.device('cpu')
    script_name = os.path.basename(__file__).split('.')[0]

    # new_3:将up变成LSTM
    script_name += "eval_rnn_pro_2_in{:d}_out{:d}_dctn{:d}".format(opt.input_n, opt.output_n, opt.all_n)
    # 返回main_in10_out25_dctn35
    print(">>> creating model")

    model_GCN = nnmodel_2.GCN(input_feature=all_n, hidden_feature=opt.linear_size, p_dropout=0.5,
                              num_stage=opt.num_stage, node_n=72)

    if is_cuda:
        model_GCN = model_GCN.to(device)

    model_gcn_path = "G:/python/Work-Skin-IMU/trained-model/GCN/ckpt_main_gcntestnnnnn_n36_out24_dctn60_best.pth.tar"
    print(">>> loading ckpt len from '{}'".format(model_gcn_path))
    if is_cuda:
        ckpt_2 = torch.load(model_gcn_path)
    else:
        ckpt_2 = torch.load(model_gcn_path, map_location='cpu')
    start_epoch = ckpt_2['epoch']
    print(">>>  start_epoch", start_epoch)
    model_GCN.load_state_dict(ckpt_2['state_dict'])
    print(">>> ckpt len loaded (epoch: {} )".format(start_epoch))
    # model_res_path = ""
    # print(">>> loading ckpt len from '{}'".format(model_res_path))

    # data loading
    print(">>> loading data")
    test_dataset = dip(path_to_data=data_path, input_n=input_n, output_n=output_n,
                       split=2)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=256,  # 128
        shuffle=False,
        num_workers=0,  # 原来写的是opt.job=10，现在因为虚拟内存不够，更改为0。
        pin_memory=True)
    model_GCN.eval()
    fig = plt.figure()
    ax = plt.gca(projection='3d')
    step = 2
    for i, (input_ori, input_acc, out_poses, out_joints) in enumerate(test_loader):
        if torch.isnan(input_ori).sum() == 0 and torch.isnan(input_acc).sum() == 0 \
                and torch.isnan(out_poses).sum() == 0 and torch.isnan(out_joints).sum() == 0:
            input_ori_acc = torch.cat([input_ori.flatten(2), input_acc.flatten(2)], dim=2)
            model_input = get_dct(input_ori_acc, input_n)

            if torch.cuda.is_available():
                model_input = model_input.to(device).float()
                input_ori = input_ori.to(device).float()
                input_acc = input_acc.to(device).float()
                out_joints = out_joints.to(device).float()
            # model要改
            # pro_out = model_pro(input_ori, input_acc)
            GCN_out = model_GCN(model_input)
            batch, frame, node, dim = out_joints.data.shape

            _, idct_m = get_dct_matrix(frame)
            idct_m = torch.from_numpy(idct_m).to(torch.float32).to(device)
            outputs_t = GCN_out.view(-1, frame).permute(1, 0)
            # 50,32*24*3
            outputs_p3d = torch.matmul(idct_m[:, 0:frame], outputs_t)
            outputs_p3d = outputs_p3d.reshape(frame, batch, node, dim).contiguous().permute(1, 0, 2, 3).contiguous()
            # batch, frame, _, _ = pro_out.data.shape
            xyz_gt = out_joints
            # xyz_pro = pro_out
            xyz_gcn = outputs_p3d[:, ::step]
            # err_1 = loss_func.position_loss(xyz_pro, xyz_gt)
            # err_2 = loss_func.position_loss(xyz_gcn, xyz_gt)
            xyz_gt = xyz_gt.cpu().data.numpy()
            # xyz_pro = xyz_pro.cpu().data.numpy()
            xyz_gcn = xyz_gcn.cpu().data.numpy()
            # print(err_1, err_2)
            input_n = input_n // step
            # 第一位作为batch
            for k in range(batch // step):
                plt.cla()  # 清除当前轴
                pig_name = "G:/python/Work-Skin-IMU/trained-model/picture/gcn/s1001a/" + "batch{:d}".format(i) + \
                           "seq{:d}".format(
                               (k + 1))
                # xyz_pred[:, :, node_ignore] = xyz_gt[:, :, node_ignore]
                figure_title = "batch:{:d},".format(i) + "seq:{},".format((k + 1))

                viz.plot_predictions_2(xyz_gt[k], xyz_gt[k], xyz_gt[k], fig, ax, figure_title, pig_name)
                # viz.plot_gt(xyz_gt[k], fig, ax, figure_title, input_n,pig_name)
                plt.pause(0.5)


def demo_residual(opt):
    input_n = opt.input_n
    output_n = opt.output_n
    all_n = input_n + output_n
    data_path = "C:/Gtrans/dataset/IMU Dataset/DIP_IMU/test/s_10/01_a.pkl"
    adjs = opt.adjs
    is_cuda = False
    device = torch.device('cpu')
    script_name = os.path.basename(__file__).split('.')[0]

    # new_3:将up变成LSTM
    script_name += "eval_rnn_pro_2_in{:d}_out{:d}_dctn{:d}".format(opt.input_n, opt.output_n, opt.all_n)
    # 返回main_in10_out25_dctn35
    print(">>> creating model")
    model_residual = residual.Seq2SeqModel(input_seq=input_n, target_seq=output_n,
                                           rnn_size=128, input_size=72, output_size=72, device=device)
    if is_cuda:
        model_residual = model_residual.to(device)

    model_gcn_path = "G:/python/Work-Skin-IMU/trained-model/residual/ckpt_main_residual.tar"
    print(">>> loading ckpt len from '{}'".format(model_gcn_path))
    if is_cuda:
        ckpt_2 = torch.load(model_gcn_path)
    else:
        ckpt_2 = torch.load(model_gcn_path, map_location='cpu')
    start_epoch = ckpt_2['epoch']
    print(">>>  start_epoch", start_epoch)
    model_residual.load_state_dict(ckpt_2['state_dict'])
    print(">>> ckpt len loaded (epoch: {} )".format(start_epoch))

    # data loading
    print(">>> loading data")
    test_dataset = dip(path_to_data=data_path, input_n=input_n, output_n=output_n,
                       split=2)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=256,  # 128
        shuffle=False,
        num_workers=0,  # 原来写的是opt.job=10，现在因为虚拟内存不够，更改为0。
        pin_memory=True)
    model_residual.eval()
    fig = plt.figure()
    ax = plt.gca(projection='3d')
    step = 2
    for i, (input_ori, input_acc, out_poses, out_joints) in enumerate(test_loader):
        if torch.isnan(input_ori).sum() == 0 and torch.isnan(input_acc).sum() == 0 \
                and torch.isnan(out_poses).sum() == 0 and torch.isnan(out_joints).sum() == 0:
            input_ori_acc = torch.cat([input_ori.flatten(2), input_acc.flatten(2)], dim=2)
            pad_idx = np.repeat([input_n - 1], output_n)
            i_idx = np.append(np.arange(0, input_n), pad_idx)
            encoder_input = input_ori_acc[:, 0:input_n - 1, :]
            decoder_input = input_ori_acc[:, i_idx, :]
            if is_cuda:
                encoder_input = encoder_input.to(device).float()
                decoder_input = decoder_input.to(device).float()
                out_joints = out_joints.to(device).float()
            y_out = model_residual(encoder_input, decoder_input)  # 32,72,50
            batch, frame, node, dim = out_joints.data.shape
            y_out = y_out.view(batch, frame, node, -1)
            xyz_gt = out_joints
            # xyz_pro = pro_out
            xyz_gcn = y_out[:, ::step]
            # err_1 = loss_func.position_loss(xyz_pro, xyz_gt)
            # err_2 = loss_func.position_loss(xyz_gcn, xyz_gt)
            xyz_gt = xyz_gt.cpu().data.numpy()
            # xyz_pro = xyz_pro.cpu().data.numpy()
            xyz_gcn = xyz_gcn.cpu().data.numpy()
            # print(err_1, err_2)
            input_n = input_n // step
            # 第一位作为batch
            for k in range(batch // step):
                plt.cla()  # 清除当前轴
                pig_name = "G:/python/Work-Skin-IMU/trained-model/picture/residual/s1001a/" + "batch{:d}".format(i) + \
                           "seq{:d}".format(
                               (k + 1))
                # xyz_pred[:, :, node_ignore] = xyz_gt[:, :, node_ignore]
                figure_title = "batch:{:d},".format(i) + "seq:{},".format((k + 1))

                viz.plot_predictions_n(xyz_gcn[k], xyz_gcn[k], xyz_gcn[k], fig, ax, figure_title, pig_name)
                # viz.plot_gt(xyz_gt[k], fig, ax, figure_title, input_n,pig_name)
                plt.pause(0.5)


def eval_total(opt):
    input_n = opt.input_n
    output_n = opt.output_n
    all_n = input_n + output_n
    adjs = opt.adjs
    is_cuda = False
    device = torch.device('cpu')
    script_name = os.path.basename(__file__).split('.')[0]


def eval_dip_pose(opt):
    input_n = opt.input_n
    output_n = opt.output_n
    all_n = input_n + output_n
    # data_path = "/data/wwu/xt/dataset/TotalCapture_Real/test/acting/s3_acting2.pkl"
    print(" torch.cuda.is_available()", torch.cuda.is_available())
    # device = torch.device("cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    adjs = opt.adjs
    is_cuda = False
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    script_name = os.path.basename(__file__).split('.')[0]

    # new_3:将up变成LSTM
    script_name += "eval_rnn_pro_2_in{:d}_out{:d}_dctn{:d}".format(opt.input_n, opt.output_n, opt.all_n)
    # 返回main_in10_out25_dctn35
    print(">>> creating model")

    model_pro = nn_pose.Predict_imu(input_frame_len=all_n, output_frame_len=all_n, input_size=72,
                                    mid_size=18, output_size=216, adjs=adjs,
                                    device=device, dropout=0.2)
    model_pro = model_pro.to(device)
    model_pro_path = "/home/xt/Skim-IMU-master/checkpoint/test/ckpt_main_6_posenew_222_in36_out24_dctn60_best.pth.tar"
    print(">>> loading ckpt len from '{}'".format(model_pro_path))
    if is_cuda:
        ckpt = torch.load(model_pro_path)
    else:
        ckpt = torch.load(model_pro_path, map_location='cpu')
    start_epoch = ckpt['epoch']
    print(">>>  start_epoch", start_epoch)
    model_pro.load_state_dict(ckpt['state_dict'])
    print(">>> ckpt len loaded (epoch: {} )".format(start_epoch))
    test_dataset = amass(path_to_data=opt.data_xt_dip_dir, input_n=input_n, output_n=output_n,
                         split=3)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=256,  # 128
        shuffle=False,
        num_workers=0,  # 原来写的是opt.job=10，现在因为虚拟内存不够，更改为0。
        pin_memory=False)
    N = 0
    eval_frame = [5, 17, 29, 35, 41, 47, 53, 59]
    test_all = torch.zeros([len(eval_frame), 4])
    # official_model_file="/data/xt/body_models/vPOSE/models/smpl/SMPL_MALE.pkl",
    #                                         smpl_folder="/data/xt/body_models/vPOSE/models"
    model_pro.eval()
    evaluator = loss_func.PoseEvaluator(official_model_file="/data/xt/body_models/vPOSE/models/smpl/SMPL_MALE.pkl",
                                        smpl_folder="/data/xt/body_models/vPOSE/models")
    for i, (input_ori, input_acc, out_poses, out_joints) in enumerate(test_loader):
        if torch.isnan(input_ori).sum() == 0 and torch.isnan(input_acc).sum() == 0 \
                and torch.isnan(out_poses).sum() == 0 and torch.isnan(out_joints).sum() == 0:
            if torch.cuda.is_available():
                # model_input = model_input.to(device).float()
                input_ori = input_ori.to(device).float()
                input_acc = input_acc.to(device).float()
                out_poses = out_poses.to(device).float()
            # model要改
            y_out = model_pro(input_ori, input_acc)
            print(y_out.shape)
            batch, frame, _, _ = y_out.data.shape
            for k in np.arange(0, len(eval_frame)):
                j = eval_frame[k]
                test_out, test_poses = y_out[:, j, :], out_poses[:, j, :]
                test_all[k] += (evaluator.eval_all(test_out.cpu(), test_poses.cpu())) * batch
            N += batch
    print((test_all / N).flatten(0))


# 一维DCT变换
def get_dct_matrix(N):
    dct_m = np.eye(N)  # 返回one-hot数组
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)  # 2/35开更
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)  # 矩阵求逆
    return dct_m, idct_m


def get_dct(out_joints, input_n):
    batch, frame, dim = out_joints.shape
    dct_m_in, _ = get_dct_matrix(frame)
    input_joints = out_joints.transpose(0, 1).reshape(frame, -1).contiguous()
    pad_idx = np.repeat([input_n - 1], frame - input_n)  # 25个(10-1)
    i_idx = np.append(np.arange(0, input_n), pad_idx)  # 前十个是输入，后二十五个是最后一个姿势
    input_dct_seq = np.matmul((dct_m_in[0:frame, :]), input_joints[i_idx, :])
    input_dct_seq = torch.as_tensor(input_dct_seq)
    input_joints = input_dct_seq.transpose(1, 0).reshape(batch, dim, frame)
    return input_joints


def get_idct(y_out, out_joints, device):
    batch, frame, node, dim = out_joints.data.shape

    _, idct_m = get_dct_matrix(frame)
    idct_m = torch.from_numpy(idct_m).to(torch.float32).to(device)
    outputs_t = y_out.view(-1, frame).transpose(1, 0)
    # 50,32*24*3
    outputs_p3d = torch.matmul(idct_m[:, 0:frame], outputs_t)
    outputs_p3d = outputs_p3d.reshape(frame, batch, node, dim).contiguous().permute(1, 0, 2, 3).contiguous()
    # 32,72,50
    pred_3d = outputs_p3d.contiguous().view(-1, 3).contiguous()
    targ_3d = out_joints.contiguous().view(-1, 3).contiguous()
    return pred_3d, targ_3d


# option = Options().parse()
# # demo_benji(option)
# # eval(option)
#
# # demo_residual(option)
# # eval_dip_pose(option)
# print()
# # con = nn.Conv1d(1024, 1024, kernel_size=9, padding='same', dilation=1)
# # x = torch.randn(256, 1024, 24)
# # y = con(x)
#
# print(y.shape)
"""
from utils.utils_math.angular import RotationRepresentation

from utils.SMPLmodel import ParametricModel
import torch
from utils.utils_math import angular as A
amass_path = "/data/xt/dataset/AMASS/train/SSM_synced/20161014_50033/punch_kick_sync_poses.npz"
cdata = numpy.load(amass_path)
print(cdata['mocap_framerate'])

import glob
import os
from tqdm import tqdm

total_path = r"/data/xt/dataset/dip/train"
out_path = r"/data/xt/dataset/AMASS_DIP_2/val"
comp_device = torch.device("cpu")
print("comp_device", comp_device)
idx = 0
subset_dir = [x for x in os.listdir(total_path)
              if os.path.isdir(total_path + '/' + x)]
for subset in subset_dir:

    # 数据输入
    # 数据的拼接
    seqs = glob.glob(os.path.join(total_path, subset, '*.pkl'))
    print('-- processing subset {:s}'.format(subset))
    for seq in tqdm(seqs):
        print(seq)
        data = pkl.load(open(seq, 'rb'), encoding='latin1')
        acc = torch.from_numpy(data['imu_acc']).float()  # n,3
        ori = torch.from_numpy(data['imu_ori']).float()  # n,3,3
        poses = torch.from_numpy(data['gt']).float()  # n,72
        print(acc.shape, ori.shape, poses.shape)
        idx += acc.shape[0]
print("idx", idx)
"""


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dim_per_head=64, dropout=0.):
        super().__init__()

        self.num_heads = num_heads
        self.scale = dim_per_head ** -0.5

        inner_dim = dim_per_head * num_heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.attend = nn.Softmax(dim=-1)

        project_out = not (num_heads == 1 and dim_per_head == dim)
        self.out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, l, d = x.shape

        '''i. QKV projection'''
        # (b,l,dim_all_heads x 3)
        qkv = self.to_qkv(x)
        # (3,b,num_heads,l,dim_per_head)
        qkv = qkv.view(b, l, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4).contiguous()
        # 3 x (1,b,num_heads,l,dim_per_head)
        q, k, v = qkv.chunk(3)
        q, k, v = q.squeeze(0), k.squeeze(0), v.squeeze(0)

        '''ii. Attention computation'''
        attn = self.attend(
            torch.matmul(q, k.transpose(-1, -2)) * self.scale
        )
        print("k.shape,k.transpose(-1,-2).shape", k.shape, k.transpose(-1, -2).shape)
        print(attn.shape)
        print((torch.matmul(q, k.transpose(-1, -2)) * self.scale).shape)
        '''iii. Put attention on Value & reshape'''
        # (b,num_heads,l,dim_per_head)
        z = torch.matmul(attn, v)
        # (b,num_heads,l,dim_per_head)->(b,l,num_heads,dim_per_head)->(b,l,dim_all_heads)
        z = z.transpose(1, 2).reshape(b, l, -1)
        # assert z.size(-1) == q.size(-1) * self.num_heads

        '''iv. Project out'''
        # (b,l,dim_all_heads)->(b,l,dim)
        out = self.out(z)
        # assert out.size(-1) == d

        return out


att = SelfAttention(dim=256, num_heads=8, dim_per_head=64, dropout=0.1)
x = torch.randn(256, 12, 256)
y = att(x)
