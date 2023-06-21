import utils.viz as viz

import torch
import torch.optim
from torch.utils.data import DataLoader
import os
from matplotlib import pyplot as plt
import numpy as np
from utils.single_total import amass_rnn as total
from utils.single_dip import amass_rnn as dip
import model.model_new as nnmodel_IPP
import model.PVRED as nnmodel_PVRED
import model.GCN as nnmodel_GCN
import model.Seq2Seq as nnmodel_residual
import model.RNN_6_pose_DIP as nn_pose
from utils.opt import Options
from utils import loss_func
from utils.amass_6_node import amass_rnn as amass


def view_all(opt):
    input_n = opt.input_n
    output_n = opt.output_n
    all_n = input_n + output_n
    dip_path = "C:/Gtrans/dataset/IMU Dataset/DIP_IMU/test/s_10/01_a.pkl"
    total_path = "C:/Gtrans/dataset/IMU Dataset/TotalCapture_Real/all/s3_rom3.pkl"
    adjs = opt.adjs
    device = torch.device('cpu')
    print(">>> creating model")
    model_IPP, model_GCN, model_residual, model_PVRED = create_model(opt, adjs, device=device)
    # test_dataset = dip(path_to_data=dip_path, input_n=input_n, output_n=output_n,
    #                    split=2)
    test_dataset = total(path_to_data=total_path, input_n=input_n, output_n=output_n,
                         split=2)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=2048,  # 128
        shuffle=False,
        num_workers=0,  # 原来写的是opt.job=10，现在因为虚拟内存不够，更改为0。
        pin_memory=True)
    fig = plt.figure(figsize=(8, 2))
    # fig = plt.figure()
    ax = plt.gca(projection='3d')
    model_IPP.eval(), model_GCN.eval(), model_residual.eval(), model_PVRED.eval()
    for i, (input_ori, input_acc, out_poses, out_joints) in enumerate(test_loader):
        if torch.isnan(input_ori).sum() == 0 and torch.isnan(input_acc).sum() == 0 \
                and torch.isnan(out_poses).sum() == 0 and torch.isnan(out_joints).sum() == 0:
            pad_idx = np.repeat([input_n - 1], output_n)
            i_idx = np.append(np.arange(0, input_n), pad_idx)
            input_ori = input_ori
            input_acc = input_acc[:, i_idx, :]
            input_ori = input_ori[:, i_idx, :]
            input_ori_acc = torch.cat([input_ori.flatten(2), input_acc.flatten(2)], dim=2)
            GCN_input = get_dct(input_ori_acc, input_n)
            PVRED_input = input_ori_acc[:, 0:input_n - 1, :].transpose(0, 1)
            target_input = input_ori_acc[:, input_n - 1:].transpose(0, 1)
            Resi_input = input_ori_acc[:, 0:input_n - 1, :]
            input_ori = input_ori[:, :input_n]
            input_acc = input_acc[:, :input_n]
            if torch.cuda.is_available():
                GCN_input = GCN_input.to(device).float()
                input_ori = input_ori.to(device).float()
                input_acc = input_acc.to(device).float()
                PVRED_input = PVRED_input.to(device).float()
                target_input = target_input.to(device).float()
                Resi_input = Resi_input.to(device).float()
                input_acc = input_acc.to(device).float()
                out_joints = out_joints.to(device).float()
            # model要改
            IPP_out = model_IPP(input_ori, input_acc)
            for idx in range(IPP_out.shape[0]):
                print("seq:", idx, torch.mean(torch.norm(IPP_out[idx] - out_joints[idx], 2, 2)))
            IPP_out = IPP_out.detach().numpy()
            outputs_enc, outputs_dec = model_PVRED(PVRED_input, target_input)
            PVRED_out = outputs_dec.transpose(0, 1).detach().numpy()
            resi_out = model_residual(Resi_input, Resi_input).detach().numpy()
            GCN_out = model_GCN(GCN_input)
            GCN_out = get_idct(GCN_out, out_joints, device)[:, input_n:].detach().numpy()
            # residual_out = model_residual(encoder_input, decoder_input)[:, ::step].detach().numpy()
            out_joints = out_joints
            xyz_gt = out_joints
            xyz_gt_out = xyz_gt[:, input_n:, ]
            batch = out_joints.shape[0]
            # # DIP_batch = [0, 6, 7, 15, 25, 26, 32, 33, 34, 36, 41, 84]
            # total_batch = [13, 14, 15, 16, 24, 34, 35, 36, 37, 38, 39, 40, 46, 47, 48, 49, 71, 72, 73, 74]
            # dip_batch = [22, 27, 33, 38, 41]
            total_batch = [1776, 1799]
            dip_batch = [33, 37, 40, 85]
            for k in total_batch:
                plt.cla()  # 清除当前轴
                pig_name = "G:/python/Work-Skin-IMU/trained-model/picture/total-joint/rom" + \
                           "/batch{:d}_seq{:d}".format(i, (k + 1))
                figure_title = "batch:{:d},".format(i) + "seq:{},".format((k + 1))
                viz.plot_predictions_n(xyz_gt[k], xyz_gt[k],
                                       xyz_gt[k], xyz_gt[k], xyz_gt[k],
                                       fig, ax, figure_title, pig_name + '_gt', pre=0)
                plt.pause(0.0001)
                plt.cla()  # 清除当前轴

                # viz.plot_predictions_n(IPP_out[k], xyz_gt[k],
                #                        IPP_out[k], IPP_out[k], IPP_out[k],
                #                        fig, ax, figure_title, pig_name + "_IPP", pre=1)
                # plt.pause(0.0001)
                # plt.cla()  # 清除当前轴
                # viz.plot_predictions_n(GCN_out[k], xyz_gt_out[k],
                #                        GCN_out[k], resi_out[k], PVRED_out[k],
                #                        fig, ax, figure_title, pig_name + "_GCN", pre=1)
                # plt.pause(0.0001)
                # plt.cla()  # 清除当前轴
                # viz.plot_predictions_n(GCN_out[k], xyz_gt_out[k],
                #                        resi_out[k], resi_out[k], PVRED_out[k],
                #                        fig, ax, figure_title, pig_name + "_resi", pre=1)
                # plt.pause(0.0001)
                # plt.cla()  # 清除当前轴
                viz.plot_predictions_n(GCN_out[k], xyz_gt_out[k],
                                       PVRED_out[k], resi_out[k], PVRED_out[k],
                                       fig, ax, figure_title, pig_name + "_PVRED", pre=1)
                plt.pause(0.0001)


def view_gt(opt):
    input_n = opt.input_n
    output_n = opt.output_n
    all_n = input_n + output_n
    dip_path = "C:/Gtrans/dataset/IMU Dataset/DIP_IMU/test/s_09/02_b.pkl"
    total_path = "C:/Gtrans/dataset/IMU Dataset/TotalCapture_Real/all/s3_freestyle3.pkl"
    adjs = opt.adjs
    device = torch.device('cpu')
    print(">>> creating model")
    test_dataset = dip(path_to_data=dip_path, input_n=input_n, output_n=output_n,
                       split=2)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=256,  # 128
        shuffle=False,
        num_workers=0,  # 原来写的是opt.job=10，现在因为虚拟内存不够，更改为0。
        pin_memory=True)
    step = 2
    fig = plt.figure(figsize=(8, 2))
    ax = plt.gca(projection='3d')
    for i, (input_ori, input_acc, out_poses, out_joints) in enumerate(test_loader):
        if torch.isnan(input_ori).sum() == 0 and torch.isnan(input_acc).sum() == 0 \
                and torch.isnan(out_poses).sum() == 0 and torch.isnan(out_joints).sum() == 0:
            xyz_gt = out_joints[:, ::2]
            batch = out_joints.shape[0]
            dip_03a_batch = [25, 26]
            for k in dip_03a_batch:
                plt.cla()  # 清除当前轴
                pig_name = "G:/python/Work-Skin-IMU/trained-model/picture/DIP-Joint/02/" + "freestyle" \
                                                                                           "seq{:d}".format(
                    (k + 1))
                # xyz_pred[:, :, node_ignore] = xyz_gt[:, :, node_ignore]
                figure_title = "batch:{:d},".format(i) + "seq:{},".format((k + 1))
                # viz.plot_predictions_n(xyz_gt[k], GCN_out[k],
                #                        residual_out[k], jinRnn_out[k],
                #                        rnngcn_out[k], rnnpro_out[k], IPP_out[k],
                #                        fig, ax, figure_title, pig_name)
                viz.plot_predictions_n(xyz_gt[k], xyz_gt[k],

                                       xyz_gt[k], xyz_gt[k], xyz_gt[k],
                                       fig, ax, figure_title, pig_name)
                plt.pause(0.01)


def create_model(opt, adjs, device):
    all_n = 60
    model_path = "G:/python/Work-Skin-IMU/trained-model/all"
    model_IPP = nnmodel_IPP.Predict_imu(input_frame_len=all_n, output_frame_len=all_n, input_size=72,
                                        mid_size=18, output_size=72, adjs=adjs,
                                        device=device, dropout=0.3)
    model_IPP_path = model_path + '/IPP_joint.tar'
    print(">>> loading ckpt len from '{}'".format(model_IPP_path))
    ckpt = torch.load(model_IPP_path, map_location='cpu')
    start_epoch = ckpt['epoch']
    print(">>>  start_epoch", start_epoch)
    model_IPP.load_state_dict(ckpt['state_dict'])
    model_GCN = nnmodel_GCN.GCN(input_feature=all_n, hidden_feature=128, p_dropout=0.5,
                                num_stage=12, node_n=72)
    model_gcn_path = model_path + '/GCN_joint.tar'
    print(">>> loading ckpt len from '{}'".format(model_gcn_path))
    ckpt = torch.load(model_gcn_path, map_location='cpu')
    start_epoch = ckpt['epoch']
    print(">>>  start_epoch", start_epoch)
    model_GCN.load_state_dict(ckpt['state_dict'])
    model_residual = nnmodel_residual.Seq2SeqModel(input_seq=opt.input_n, target_seq=opt.output_n,
                                                   rnn_size=128, input_size=72, output_size=72, device=device)
    model_residual_path = model_path + '/residual_joint.tar'
    print(">>> loading ckpt len from '{}'".format(model_residual_path))
    ckpt = torch.load(model_residual_path, map_location='cpu')
    start_epoch = ckpt['epoch']
    print(">>>  start_epoch", start_epoch)
    model_residual.load_state_dict(ckpt['state_dict'])
    model_PVRED = nnmodel_PVRED.Encoder_Decoder(input_size=72, hidden_size=1024, num_layer=1, rnn_unit='gru',
                                                residual=True, out_dropout=0.3, std_mask=True, veloc=True,
                                                pos_embed=True
                                                , pos_embed_dim=96, device=device)
    model_PVRED_path = model_path + '/PVRED_joint.tar'
    print(">>> loading ckpt len from '{}'".format(model_PVRED_path))
    ckpt = torch.load(model_PVRED_path, map_location='cpu')
    start_epoch = ckpt['epoch']
    print(">>>  start_epoch", start_epoch)
    model_PVRED.load_state_dict(ckpt['state_dict'])
    return model_IPP, model_GCN, model_residual, model_PVRED


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
    outputs_t = y_out.view(-1, frame).permute(1, 0)
    # 50,32*24*3
    outputs_p3d = torch.matmul(idct_m[:, 0:frame], outputs_t)
    outputs_p3d = outputs_p3d.reshape(frame, batch, node, dim).contiguous().permute(1, 0, 2, 3).contiguous()
    return outputs_p3d


# def view_model(opt):
#     input_n = opt.input_n
#     output_n = opt.output_n
#     all_n = input_n + output_n
#     dip_path = "C:/Gtrans/dataset/IMU Dataset/DIP_IMU/test/s_10/01_a.pkl"
#     total_path = "C:/Gtrans/dataset/IMU Dataset/TotalCapture_Real/all/s3_freestyle3.pkl"
#     print(" torch.cuda.is_available()", torch.cuda.is_available())
#     # device = torch.device("cuda:{gpu}" if torch.cuda.is_available() else "cpu")
#     # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     adjs = opt.adjs
#     is_cuda = False
#     # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     device = torch.device('cpu')
#     script_name = os.path.basename(__file__).split('.')[0]
#
#     # new_3:将up变成LSTM
#     script_name += "eval_rnn_pro_2_in{:d}_out{:d}_dctn{:d}".format(opt.input_n, opt.output_n, opt.all_n)
#     # 返回main_in10_out25_dctn35
#     print(">>> creating model")
#
#     model_pro = nnmodel_IPP.Predict_imu(input_frame_len=all_n, output_frame_len=all_n, input_size=72,
#                                         mid_size=18, output_size=72, adjs=adjs,
#                                         device=device, dropout=0.2)
#     # model_GCN = nnmodel_2.GCN(input_feature=all_n, hidden_feature=128, p_dropout=0.5,
#     #                           num_stage=12, node_n=72)
#
#     # if is_cuda:
#     #     model_GCN = model_GCN.to(device)
#     model_pro = model_pro.to(device)
#
#     model_pro_path = "G:/python/Work-Skin-IMU/trained-model/best/total_model.tar"
#     # "ckpt_main_rnn_gcn_pron222_in36_out24_dctn60_best.pth" \
#
#     print(">>> loading ckpt len from '{}'".format(model_pro_path))
#     if is_cuda:
#         ckpt = torch.load(model_pro_path)
#     else:
#         ckpt = torch.load(model_pro_path, map_location='cpu')
#     start_epoch = ckpt['epoch']
#     print(">>>  start_epoch", start_epoch)
#     model_pro.load_state_dict(ckpt['state_dict'])
#     print(">>> ckpt len loaded (epoch: {} )".format(start_epoch))
#     # print(model_pro)
#     test_dataset = dip(path_to_data=dip_path, input_n=input_n, output_n=output_n,
#                        split=2)
#     test_loader = DataLoader(
#         dataset=test_dataset,
#         batch_size=256,  # 128
#         shuffle=False,
#         num_workers=0,  # 原来写的是opt.job=10，现在因为虚拟内存不够，更改为0。
#         pin_memory=True)
#     model_pro.eval()
#     fig = plt.figure()
#     ax = plt.gca(projection='3d')
#     for i, (input_ori, input_acc, out_poses, out_joints) in enumerate(test_loader):
#         if torch.isnan(input_ori).sum() == 0 and torch.isnan(input_acc).sum() == 0 \
#                 and torch.isnan(out_poses).sum() == 0 and torch.isnan(out_joints).sum() == 0:
#             input_ori_acc = torch.cat([input_ori.flatten(2), input_acc.flatten(2)], dim=2)
#
#             if torch.cuda.is_available():
#                 input_ori = input_ori.to(device).float()
#                 input_acc = input_acc.to(device).float()
#                 out_joints = out_joints.to(device).float()
#             # model要改
#             batch, frame, node, dim = out_joints.data.shape
#             pro_out = model_pro(input_ori, input_acc)
#             err = loss_func.position_loss(pro_out, out_joints=out_joints)
#             print("err", err)
#             step = 2
#             input_n = input_n // step
#             xyz_gt = out_joints[:, ::step]
#             xyz_pro = pro_out[:, ::step]
#             xyz_gt = xyz_gt.cpu().data.numpy()
#             xyz_pro = xyz_pro.cpu().data.numpy()
#             # 第一位作为batch
#             DIP_batch = [0, 6, 7, 15, 16, 25, 26, 32, 33, 34, 36, 39, 41, 48, 84]
#             for k in DIP_batch:
#                 plt.cla()  # 清除当前轴
#                 pig_name = "G:/python/Work-Skin-IMU/trained-model/picture/DIP-Joint/Only2/" + 's1001a' + \
#                            "seq{:d}".format(
#                                (k + 1))
#                 # xyz_pred[:, :, node_ignore] = xyz_gt[:, :, node_ignore]
#                 figure_title = "batch:{:d},".format(i) + "seq:{},".format((k + 1))
#
#                 viz.plot_predictions_3(xyz_gt[k], xyz_pro[k], xyz_pro[k], fig, ax, figure_title, pig_name)
#                 # viz.plot_gt(xyz_gt[k], fig, ax, figure_title, input_n,pig_name)
#                 plt.pause(0.1)

option = Options().parse()
# view_model(option)
view_all(option)
# view_gt(option)
