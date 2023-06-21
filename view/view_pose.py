import torch
import numpy as np
import pickle
import utils.viz as viz
from utils.utils_math import angular as A
from utils.SMPLmodel import ParametricModel as smplmodel
import torch
import smplx
import torch.optim
from torch.utils.data import DataLoader
import os
from matplotlib import pyplot as plt
from utils.single_total import amass_rnn as total
from utils.single_dip import amass_rnn as dip
import model.RNN_6_pose_DIP as nnmodel_IPP
import model.RNN_best as nnmodel_rnn
import model.GCN as nnmodel_GCN
import model.Seq2Seq as nnmodel_residual
import model.pvred_enc as nnmodel_PVRED
import model.Anastudy.jinRNN as nnmodel_jinRNN
import model.Anastudy.rnn_pro as nnmodel_rnnpro
import model.Anastudy.rnn_gcn as nnmodel_rnngcn
import utils.loss_func as loss_func
from utils.opt import Options
from utils.single_total import amass_rnn as total
from utils.single_dip import amass_rnn as dip
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from os import path as osp
from torch.utils.data import DataLoader
import trimesh
from body_visualizer.tools.vis_tools import colors
from body_visualizer.mesh.mesh_viewer import MeshViewer
from body_visualizer.mesh.sphere import points_to_spheres
from body_visualizer.tools.vis_tools import show_image
from body_visualizer.tools.vis_tools import imagearray2file
# 可视化部件
import cv2
import trimesh
from body_visualizer.tools.vis_tools import colors
from body_visualizer.mesh.mesh_viewer import MeshViewer
from body_visualizer.mesh.sphere import points_to_spheres
from body_visualizer.tools.vis_tools import show_image
from human_body_prior.body_model.body_model import BodyModel

imw, imh = 720, 720
mv = MeshViewer(width=imw, height=imh, use_offscreen=True)
import vctoolkit as vc
import vctoolkit.viso3d as vo3d

imu_mask = [7, 8, 11, 12, 0, 2]

import torch
# from human_body_prior.body_model.body_model import BodyModel
# # 可视化部件
# import trimesh
# from body_visualizer.tools.vis_tools import colors
# from body_visualizer.mesh.mesh_viewer import MeshViewer
# from body_visualizer.mesh.sphere import points_to_spheres
# from body_visualizer.tools.vis_tools import show_image

import vctoolkit  # 安装好了


class Ax3DPose(object):
    def __init__(self, ax):
        """
        Create a 3d pose visualizer that can be updated with new poses.
        Args
          ax: 3d axis to plot the 3d pose on ：ax = plt.gca(projection='3d')
          lcolor: String. Colour for the left part of the body
          rcolor: String. Colour for the right part of the body
        """
        self.ax = ax
        self.plots = []
        # self.body_mash = []
        vertices = torch.randn(6890, 3)
        self.body_mesh = trimesh.Trimesh(vertices=c2c(vertices))
        # mv.set_static_meshes([self.body_mesh])
        # body_image = mv.render(render_wireframe=False)
        # img = body_image.astype(np.uint8)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # self.ax.imshow(img)

    def update(self, vertices, faces):
        vertex_colors = np.tile(colors['grey'], (6890, 1))
        self.body_mesh = trimesh.Trimesh(vertices=c2c(vertices), faces=faces,
                                         vertex_colors=vertex_colors)
        mv.set_static_meshes([self.body_mesh])
        body_image = mv.render(render_wireframe=False)
        # plt.savefig(pig_name + ' frame{:d}'.format(i + 1) + ".png")
        img = body_image.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img


def plot_predictions(allmesh, frame, ax, fig, figure_title, pig_name):
    # === Plot and animate ===
    # Plot the prediction
    meshes_gt, meshes_IPP, meshes_GCN, meshes_Res, meshes_jinRNN, meshes_rnnpro, meshes_rnngcn = allmesh
    for i in range(frame):
        ax.imshow(meshes_gt[i])
        ax.imshow(meshes_IPP[i])
        # ax.imshow(meshes_GCN[i])
        # ax.imshow(meshes_Res[i])
        # ax.imshow(meshes_jinRNN[i])
        # ax.imshow(meshes_rnnpro[i])
        # ax.imshow(meshes_rnngcn[i])
        ax.set_title(figure_title + ' frame:{:d}'.format(i + 1), loc="left")
        # plt.savefig(pig_name + ' frame{:d}'.format(i + 1) + ".png")
        plt.axis('off')
        plt.show(block=False)
        fig.canvas.draw()
        plt.pause(0.001)


def view(opt):
    input_n = opt.input_n
    output_n = opt.output_n
    all_n = input_n + output_n
    adjs = opt.adjs
    device = torch.device('cpu')
    dip_path = "C:/Gtrans/dataset/IMU Dataset/DIP_IMU/test/s_10/03_a.pkl"
    total_path = "c:/Gtrans/dataset/IMU Dataset/TotalCapture_Real/all/s3_rom3.pkl"
    bm_fname = "C:/Gtrans/body_models/vPOSE/models/smpl/SMPL_MALE.pkl"
    with open(bm_fname, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    faces = data['f']
    model_IPP, model_GCN, model_residual, model_PVRED = create_model(opt, adjs,
                                                                     device=device)
    # test_dataset = dip(path_to_data=dip_path, input_n=input_n, output_n=output_n,
    #                    split=2)
    test_dataset = total(path_to_data=total_path, input_n=input_n, output_n=output_n,
                         split=2)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=400,  # 128
        shuffle=False,
        num_workers=0,  # 原来写的是opt.job=10，现在因为虚拟内存不够，更改为0。
        pin_memory=True)
    fig = plt.figure(figsize=(8, 3))
    ax = plt.gca()
    model_IPP.eval(), model_GCN.eval(), model_residual.eval(), model_PVRED.eval()
    for i, (input_ori, input_acc, out_poses, out_joints) in enumerate(test_loader):
        if i != 4:
            continue
        if torch.isnan(input_ori).sum() == 0 and torch.isnan(input_acc).sum() == 0 \
                and torch.isnan(out_poses).sum() == 0 and torch.isnan(out_joints).sum() == 0:
            batch = out_joints.shape[0]
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
            print("all",
                  torch.mean(torch.norm(IPP_out.reshape(-1, 9) - out_poses.reshape(-1, 9), 2, 1)))
            for idx in range(IPP_out.shape[0]):
                print("seq:", idx,
                      torch.mean(torch.norm(IPP_out[idx].reshape(-1, 9) - out_poses[idx].reshape(-1, 9), 2, 1)))
            IPP_out = IPP_out
            outputs_enc, outputs_dec = model_PVRED(PVRED_input, target_input)
            PVRED_out = outputs_dec.transpose(0, 1).reshape(batch, output_n, 24, 3, 3)
            resi_out = model_residual(Resi_input, Resi_input).reshape(batch, output_n, 24, 3, 3)
            GCN_out = model_GCN(GCN_input)
            GCN_out = get_idct(GCN_out, out_joints, device)[:, input_n:].reshape(batch, output_n, 24, 3, 3)
            batch = out_poses.shape[0]
            dip_batch = [176, 199]
            for k in dip_batch:
                pig_name = "G:/python/Work-Skin-IMU/trained-model/picture/total_pose/rom/" + \
                           "seq{:d}".format((k + 1))
                figure_title = "batch:{:d},".format(i) + "seq:{},".format((k + 1))
                print(figure_title)
                plt.cla()  # 清除当前轴
                gt_colors = np.tile(colors['grey'], (6890, 1))
                pre_colors = np.tile(colors['grey'], (6890, 1))
                ori_colors = np.tile("#354E87", (6890, 1))
                showallmesh(out_poses[k], faces, ax, fig, "gt, " + figure_title, pig_name + 'gt', gt_colors)
                plt.pause(0.0001)
                plt.cla()  # 清除当前轴
                showallmesh(IPP_out[k], faces, ax, fig, 'IPP, ' + figure_title, pig_name + 'IPP', pre_colors)
                plt.pause(0.0001)
                plt.cla()  # 清除当前轴
                showallmesh(GCN_out[k], faces, ax, fig, 'GCN, ' + figure_title, pig_name + 'GCN', pre_colors)
                plt.pause(0.0001)
                plt.cla()  # 清除当前轴
                showallmesh(resi_out[k], faces, ax, fig, 'Res, ' + figure_title, pig_name + 'Res', pre_colors)
                plt.pause(0.0001)
                plt.cla()  # 清除当前轴
                showallmesh(PVRED_out[k], faces, ax, fig, 'PVRED, ' + figure_title, pig_name + 'PVRED', pre_colors)
                plt.pause(0.0001)


def showallmesh(allmesh, faces, ax, fig, figure_title, pig_name, vertex_colors):
    pose = allmesh
    batch = pose.shape[0]
    meshes = []
    _, _, vertice_IPP = posetomesh(pose)
    for i in range(batch):
        body_mesh_gt = trimesh.Trimesh(vertices=c2c(vertice_IPP[i]), faces=faces,
                                       vertex_colors=vertex_colors)
        mv.set_dynamic_meshes(
            [body_mesh_gt])
        body_image = mv.render(render_wireframe=False)
        img = body_image.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        meshes.append(img)
    for i in range(batch):
        ax.imshow(meshes[i])
        plt.axis('off')
        ax.set_title(figure_title + ' frame:{:d}'.format(i + 1), loc="left")
        plt.savefig(pig_name + ' frame{:d}'.format(i + 1) + ".png")
        plt.show(block=False)
        fig.canvas.draw()
        plt.pause(0.001)


def posetomesh(pose, smpl_folder="C:/Gtrans/body_models/vPOSE/models"):
    batch, node, _, _ = pose.shape
    pose = A.rotation_matrix_to_axis_angle(pose).view(batch, node, 3).view(batch, -1)
    model_type = 'smpl'
    ext = 'pkl'
    num_expression_coeffs = 10
    use_face_contour = False
    n_frames = pose.shape[0]
    gender = 'male'
    num_betas = 10
    betas = np.zeros(num_betas)
    root_orient = pose[:, :3]
    body_pose = pose[:, 3:72]
    betas = torch.Tensor(np.repeat(betas[:num_betas][np.newaxis], repeats=n_frames, axis=0))

    model = smplx.create(model_path=smpl_folder, model_type=model_type,
                         gender=gender, num_betas=num_betas, use_face_contour=use_face_contour,
                         num_expression_coeffs=num_expression_coeffs,
                         ext=ext)
    expression = torch.randn(
        [1, model.num_expression_coeffs], dtype=torch.float32)
    output = model(global_orient=root_orient, betas=betas,
                   body_pose=body_pose, expression=expression,
                   return_verts=True, return_full_pose=True)
    joints = output.joints.detach()
    poses = output.full_pose.detach()
    joint = joints.contiguous()
    vertices = output.vertices.detach()
    return poses, joint, vertices


def create_model(opt, adjs, device):
    all_n = 60
    model_path = "G:/python/Work-Skin-IMU/trained-model/all"
    model_IPP = nnmodel_IPP.Predict_imu(input_frame_len=opt.input_n, output_frame_len=all_n, input_size=72,
                                        mid_size=18, output_size=216, adjs=adjs,
                                        device=device, dropout=0.3)
    model_IPP_path = model_path + "/IPP_pose.tar"
    print(">>> loading ckpt len from '{}'".format(model_IPP_path))
    ckpt = torch.load(model_IPP_path, map_location='cpu')
    start_epoch = ckpt['epoch']
    print(">>>  start_epoch", start_epoch)
    model_IPP.load_state_dict(ckpt['state_dict'])
    model_GCN = nnmodel_GCN.GCN(input_feature=all_n, hidden_feature=opt.linear_size, p_dropout=0.5,
                                num_stage=opt.num_stage, node_n=216)
    model_gcn_path = model_path + '/GCN_pose.tar'
    print(">>> loading ckpt len from '{}'".format(model_gcn_path))
    ckpt = torch.load(model_gcn_path, map_location='cpu')
    start_epoch = ckpt['epoch']
    print(">>>  start_epoch", start_epoch)
    model_GCN.load_state_dict(ckpt['state_dict'])
    model_residual = nnmodel_residual.Seq2SeqModel(input_seq=36, target_seq=24, output_size=216,
                                                   rnn_size=128, input_size=72, device=device)
    model_residual_path = model_path + '/residual_pose.tar'
    print(">>> loading ckpt len from '{}'".format(model_residual_path))
    ckpt = torch.load(model_residual_path, map_location='cpu')
    start_epoch = ckpt['epoch']
    print(">>>  start_epoch", start_epoch)
    model_residual.load_state_dict(ckpt['state_dict'])
    model_PVRED = nnmodel_PVRED.Encoder_Decoder(input_size=216, hidden_size=1024, num_layer=1, rnn_unit='gru',
                                                residual=True, out_dropout=0.3, std_mask=True, veloc=True,
                                                pos_embed=True
                                                , pos_embed_dim=96, device=device)
    model_PVRED_path = model_path + '/PVRED_pose.tar'
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
    batch, frame, node, _ = out_joints.data.shape

    _, idct_m = get_dct_matrix(frame)
    idct_m = torch.from_numpy(idct_m).to(torch.float32).to(device)
    outputs_t = y_out.view(-1, frame).permute(1, 0)
    # 50,32*24*3
    outputs_p3d = torch.matmul(idct_m[:, 0:frame], outputs_t)
    outputs_p3d = outputs_p3d.reshape(frame, batch, node, -1).contiguous().permute(1, 0, 2, 3).contiguous()
    return outputs_p3d


option = Options().parse()
# view_model(option)
view(option)
