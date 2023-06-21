from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import print_function, absolute_import, division

import glob
import os

import pickle as pkl

import numpy as np
import smplx

import torch

from opt import Options
from load_DIP_Model import loadDIP_nn
from tqdm import tqdm
import utils.utils_math.angular as an
import utils.SMPLmodel as smplmodel

"""

合成数据集

#            head spine，左手臂，左手腕，左膝盖，左脚踝，右手臂，右手腕，右膝盖，右脚踝
VERTEX_IDS = [411, 3021, 1618, 2243, 1146, 3198, 5208, 5560, 4500, 6610]
# 1962:左手腕 ，5431：右手腕，1096：左膝盖，4583:右膝盖412：head，3021：spine
VERTEX_IDS = [1962, 5431, 1096, 4583, 412, 3021]
"""
vi_mask = torch.tensor([1961, 5424, 1176, 4662, 411, 3021])
ji_mask = torch.tensor([18, 19, 4, 5, 15, 0])
# TODO
TARGET_FPS = 60

# TODO
# Please modify here to specify which SMPL joints to use
SMPL_IDS = [0, 15, 19, 21, 18, 20, 5, 4, 7, 8]
amass_data = ['SFU']


def Sys_amass(opt):
    smooth_n = 4

    def _syn_acc(v):
        r"""
        Synthesize accelerations from vertex positions.
        """
        mid = smooth_n // 2
        acc = torch.stack([(v[i] + v[i + 2] - 2 * v[i + 1]) * 3600 for i in range(0, v.shape[0] - 2)])
        acc = torch.cat((torch.zeros_like(acc[:1]), acc, torch.zeros_like(acc[:1])))
        if mid != 0:
            acc[smooth_n:-smooth_n] = torch.stack(
                [(v[i] + v[i + smooth_n * 2] - 2 * v[i + smooth_n]) * 3600 / smooth_n ** 2
                 for i in range(0, v.shape[0] - smooth_n * 2)])
        return acc

    amass_dataset_path = r"/data/xt/dataset/AMASS/train"
    output_dataset_path = r"/data/xt/dataset/AMASS_DIP_2/train"
    comp_device = torch.device("cuda:0")
    idx = 0
    for ds_name in amass_data:

        print('\rReading', ds_name)
        for npz_fname in tqdm(glob.glob(os.path.join(amass_dataset_path, ds_name, '*/*_poses.npz'))):
            print(npz_fname)
            try:
                cdata = np.load(npz_fname)
            except:
                continue

            framerate = int(cdata['mocap_framerate'])
            if framerate == 120:
                step = 2
            elif framerate == 60 or framerate == 59:
                step = 1
            else:
                print("mocap_framerate不符合")
                continue
            num_frame = cdata['poses'].shape[0]
            print("num_frame", num_frame)
            if num_frame < 600:
                continue
            data_pose = cdata['poses'][::step].astype(np.float32)
            data_trans = cdata['trans'][::step].astype(np.float32)
            data_beta = cdata['betas'][:10]
            dmpls = cdata['dmpls']
            tran = torch.tensor(np.asarray(data_trans, np.float32))
            pose = torch.tensor(np.asarray(data_pose, np.float32)).view(-1, 52, 3)
            pose[:, 23] = pose[:, 37]  # right hand
            pose = pose[:, :24].clone()  # only use body
            amass_rot = torch.tensor([[[1, 0, 0], [0, 0, 1], [0, -1, 0.]]])
            tran = amass_rot.matmul(tran.unsqueeze(-1)).view_as(tran)
            pose[:, 0] = an.rotation_matrix_to_axis_angle(
                amass_rot.matmul(an.axis_angle_to_rotation_matrix(pose[:, 0])))
            poses = pose.clone().flatten(1)
            n_frames = poses.shape[0]
            num_betas = 10
            num_dmpls = 8
            body_parms = {
                'root_orient': torch.Tensor(poses[:, :3]).to(comp_device),  # controls the global root orientation
                'body_pose': torch.Tensor(poses[:, 3:72]).to(comp_device),  # controls the body
                'trans': torch.Tensor(tran).to(comp_device),  # controls the global body position
                'betas': torch.Tensor(np.repeat(data_beta[:num_betas][np.newaxis], repeats=n_frames, axis=0)).to(
                    comp_device),  # controls the body shape. Body shape is static
                'dmpls': torch.Tensor(dmpls[:, :num_dmpls]).to(comp_device)  # controls soft tissue dynamics
            }
            smpl_folder = "/data/xt/body_models/vPOSE/models"
            # smpl_folder = "C:/Gtrans/body_models/vPOSE/models"

            model_type = 'smpl'
            ext = 'pkl'
            num_expression_coeffs = 10
            use_face_contour = False

            model = smplx.create(model_path=smpl_folder, model_type=model_type,
                                 gender='male', num_betas=num_betas, use_face_contour=use_face_contour,
                                 num_expression_coeffs=num_expression_coeffs,
                                 ext=ext).to(comp_device)
            expression = torch.randn(
                [1, model.num_expression_coeffs], dtype=torch.float32)
            output = model(global_orient=body_parms['root_orient'], betas=body_parms['betas'],
                           body_pose=body_parms['body_pose'], trans=body_parms['trans'], expression=expression,
                           return_verts=True)

            vertices = output.vertices.detach().cpu()
            joints = output.joints.detach().cpu().squeeze()
            body_pose = output.body_pose.detach().cpu().squeeze()
            global_orient = output.global_orient.detach().cpu().squeeze()
            n, b, dim = joints.shape
            a_global = torch.cat([global_orient, body_pose], dim=1).reshape(n, 24, 3)  # 得到关节的角度

            acc = _syn_acc(vertices[:, vi_mask])  # N, 6, 3

            ori = a_global[:, ji_mask]
            joint = joints[:, :24].contiguous()
            num_frames = len(joint)
            seq_len = 60
            fs = np.arange(0, num_frames - seq_len + 1)  # 窗口滑动
            fs_sel = fs
            for i in np.arange(seq_len - 1):
                fs_sel = np.vstack((fs_sel, fs + i + 1))  # 沿着竖直方向将矩阵堆叠起来将fs_sel和fs+i+1放到一起
            fs_sel = fs_sel.transpose()  # eg.((35, 1704)变成(1704, 35))
            amass_acc_sam = acc[fs_sel, :].numpy()
            amass_ori_sam = ori[fs_sel, :].numpy()
            amass_poses_sam = poses[fs_sel, :].numpy()
            amass_joints_sam = joint[fs_sel, :].numpy()
            # 将数据拼接到一起
            print(amass_acc_sam.shape, amass_ori_sam.shape, amass_poses_sam.shape, amass_joints_sam.shape)
            for a in range(amass_acc_sam.shape[0]):
                idx += 1
                print(a, idx)
                data_out = {'ori': amass_ori_sam[a], 'acc': amass_acc_sam[a], 'poses': amass_poses_sam[a],
                            'joints': amass_joints_sam[a]}
                # 这里创建数据输出文件夹
                out_dir = os.path.join(output_dataset_path, ds_name + str(idx) + '.pkl')
                with open(out_dir, 'wb') as fout:
                    pkl.dump(data_out, fout)


def sys():
    amass_dataset_path = r"/data/xt/dataset/AMASS/train"
    output_dataset_path = r"/data/xt/dataset/AMASS_DIP_2/train"
    # amass_subsets = [x for x in os.listdir(amass_dataset_path)
    #                 if os.path.isdir(amass_dataset_path + '/' + x)]
    # Choose the device to run the body model on.s
    comp_device = torch.device("cuda:0")
    print("comp_device", comp_device)
    subset_dir = [x for x in os.listdir(amass_dataset_path)
                  if os.path.isdir(amass_dataset_path + '/' + x)]
    acc_in, ori_in, poses_all, joints_all = [], [], [], []
    print(subset_dir)
    # 之后改为for循环
    for subset in subset_dir:
        # subset = subset_dir[0]
        # 数据输入
        subsubset_dir = os.path.join(amass_dataset_path, subset)
        subsubsets = [x for x in os.listdir(subsubset_dir)
                      if os.path.isdir(subsubset_dir + '/' + x)]
        print(subsubsets)
        # 之后改为for循环
        for subsub in subsubsets:
            # subsub = subsubsets[0]
            seqs = glob.glob(os.path.join(subsubset_dir, subsub, '*.npz'))
            print('-- processing subset {:s}'.format(subsub))
            # main loop to process each sequence
            for seq in tqdm(seqs):
                # read data
                # 跳过shape.npz
                if os.path.basename(seq) == 'shape.npz':
                    continue

                bdata = np.load(seq)
                print(bdata['poses'].shape)
                trans = bdata['trans']  # (n,3)
                if len(trans) < 500:
                    continue
                gender = str(bdata['gender'].astype(str))
                mocap_framerate = bdata['mocap_framerate']  # 120
                betas = bdata['betas']
                dmpls = bdata['dmpls']  # (n,8)
                poses = bdata['poses'].reshape(-1, 52, 3)  # (n,156)
                poses[:, 23] = poses[:, 37]  # right hand
                poses = poses[:, :24].clone().flatten(1)
                print("poses", poses.shape)
                # 将120fps转化为60fps
                # poses = interpolation_integer(poses, mocap_framerate)
                # poses = np.array(poses)
                #
                if mocap_framerate % TARGET_FPS == 0:
                    n_tmp = int(mocap_framerate / TARGET_FPS)
                    trans = trans[::n_tmp]
                    dmpls = dmpls[::n_tmp]
                    poses = poses[::n_tmp]
                n_frames = trans.shape[0]

                num_betas = 10
                num_dmpls = 8
                body_parms = {
                    'root_orient': torch.Tensor(poses[:, :3]).to(comp_device),  # controls the global root orientation
                    'body_pose': torch.Tensor(poses[:, 3:72]).to(comp_device),  # controls the body
                    'trans': torch.Tensor(trans).to(comp_device),  # controls the global body position
                    'betas': torch.Tensor(np.repeat(betas[:num_betas][np.newaxis], repeats=n_frames, axis=0)).to(
                        comp_device),  # controls the body shape. Body shape is static
                    'dmpls': torch.Tensor(dmpls[:, :num_dmpls]).to(comp_device)  # controls soft tissue dynamics
                }
                smpl_folder = "/data/xt/body_models/vPOSE/models"
                # smpl_folder = "C:/Gtrans/body_models/vPOSE/models"

                model_type = 'smpl'
                ext = 'pkl',
                num_expression_coeffs = 10,
                use_face_contour = False

                model = smplx.create(model_path=smpl_folder, model_type=model_type,
                                     gender=gender, num_betas=num_betas, use_face_contour=use_face_contour,
                                     num_expression_coeffs=num_expression_coeffs,
                                     ext=ext).to(comp_device)
                expression = torch.randn(
                    [1, model.num_expression_coeffs], dtype=torch.float32)
                output = model(global_orient=body_parms['root_orient'], betas=body_parms['betas'],
                               body_pose=body_parms['body_pose'], trans=body_parms['trans'], expression=expression,
                               return_verts=True)
                """
                output一共n个部分，vertices(n,6890,3)得到皮肤表面的6890个mish的三维空间位置，joints(n,45,3)45个关节的三维位置，body_pose(n,69),
                betas(n,10),global_orient(n,3)
                """

                vertices = output.vertices.detach().cpu().numpy().squeeze()
                joints = output.joints.detach().cpu().numpy().squeeze()
                body_pose = output.body_pose.detach().cpu().numpy().squeeze()
                global_orient = output.global_orient.detach().cpu().numpy().squeeze()
                n, b, dim = joints.shape
                a_global = np.append(global_orient, body_pose, axis=1).reshape(n, 24, 3)  # 得到关节的角度
                orientation = a_global[:, ji_mask]
                print(orientation.shape)
                vertex = vertices[:, vi_mask]
                print(vertex.shape)
                joints_pos = joints[:, :24, :]
                poses = poses[:, :72]
                # 将pose和joints都输入进去，主要是为了得到数据一样长的
                poses, joints_pos, orientation, acceleration = get_ori_acc(poses=poses, joints_pos=joints_pos,
                                                                           orientation=orientation, vertex=vertex,
                                                                           frame_rate=TARGET_FPS)
                # 在这里将数据堆叠起来，50帧的窗口滑动

                acceleration = np.array(acceleration)
                orientation = np.array(orientation)
                joints_pos = np.array(joints_pos)
                print("shape", acceleration.shape, orientation.shape, joints_pos.shape, poses.shape)
                acc_in.append(acceleration)
                ori_in.append(orientation)
                poses_all.append(poses)
                joints_all.append(joints_pos)


def Sys_dip(opt):
    DIP_dataset_path = r"C:\Gtrans\dataset\SYS_DIP\DIP\train"
    output_dataset_path = r"C:/Gtrans/dataset/SYS_DIP/train"
    loadDIP_nn(DIP_dataset_path, output_dataset_path)


def Sys_dip_path():
    DIP_path = r"/data/xt/dataset/DIP_IMU_and_Others/DIP_IMU_nn/imu_own_validation.npz"
    out_path = r"/data/xt/dataset/DIP_IMU_AMASS/val"
    comp_device = torch.device("cpu")
    if isinstance(DIP_path, str):
        data_dict = dict(np.load(DIP_path, allow_pickle=True))
    elif isinstance(DIP_path, dict):
        data_dict = DIP_path
    else:
        raise Exception("Data type isn't recognized.")
    sample_poses = data_dict['smpl_pose']
    sample_acc = data_dict['acceleration']
    sample_ori = data_dict['orientation']
    i = 0
    file_id = data_dict.get('file_id', None)
    data_id = data_dict.get('data_id', None)
    for i in range(len(sample_acc)):
        acc = torch.tensor(sample_acc[i])
        poses = torch.FloatTensor(sample_poses[i])
        ori = torch.tensor(sample_ori[i])
        for _ in range(4):
            acc[1:].masked_scatter_(torch.isnan(acc[1:]), acc[:-1][torch.isnan(acc[1:])])
            ori[1:].masked_scatter_(torch.isnan(ori[1:]), ori[:-1][torch.isnan(ori[1:])])
            acc[:-1].masked_scatter_(torch.isnan(acc[:-1]), acc[1:][torch.isnan(acc[:-1])])
            ori[:-1].masked_scatter_(torch.isnan(ori[:-1]), ori[1:][torch.isnan(ori[:-1])])

        acc, ori, poses = acc[6:-6], ori[6:-6], poses[6:-6]
        n_frames = poses.shape[0]
        smpl_folder = "/data/xt/body_models/vPOSE/models"

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
        poses = poses[:, :72]
        print(joints_pos.shape)
        num_frames = len(joints_pos)
        seq_len = 100
        fs = np.arange(0, num_frames - seq_len + 1)  # 窗口滑动
        fs_sel = fs
        for i in np.arange(seq_len - 1):
            fs_sel = np.vstack((fs_sel, fs + i + 1))  # 沿着竖直方向将矩阵堆叠起来将fs_sel和fs+i+1放到一起
        fs_sel = fs_sel.transpose()  # eg.((35, 1704)变成(1704, 35))
        amass_acc_sam = acc[fs_sel, :].numpy()
        amass_ori_sam = ori[fs_sel, :].numpy()
        amass_poses_sam = poses[fs_sel, :].numpy()
        amass_joints_sam = joints_pos[fs_sel, :]
        # 将数据拼接到一起

        for a in range(amass_acc_sam.shape[0]):
            i += 1
            data_out = {'ori': amass_ori_sam[a], 'acc': amass_acc_sam[a], 'poses': amass_poses_sam[a],
                        'joints': amass_joints_sam[a]}
            # 这里创建数据输出文件夹
            out_dir = os.path.join(out_path, "dip" + str(i) + '.pkl')
            with open(out_dir, 'wb') as fout:
                pkl.dump(data_out, fout)


def Systotal():
    total_path = r"/data/xt/dataset/TotalCapture_Real"
    out_path = r"/data/xt/dataset/AMASS_DIP_2/val"
    comp_device = torch.device("cpu")
    print("comp_device", comp_device)
    seqs = glob.glob(os.path.join(total_path, '*.pkl'))
    i = 0
    for seq in tqdm(seqs):
        print(seq)
        data = pkl.load(open(seq, 'rb'), encoding='latin1')
        ori = torch.from_numpy(data['ori']).float()
        acc = torch.from_numpy(data['acc']).float()
        pose = torch.from_numpy(data['gt']).float()
        print(acc.shape, ori.shape, pose.shape)
        min_num = min(acc.shape[0], pose.shape[0])
        acc = acc[:min_num]
        ori = ori[:min_num]
        pose = pose[:min_num]
        smpl_folder = "/data/xt/body_models/vPOSE/models"
        # smpl_folder = "C:/Gtrans/body_models/vPOSE/models"

        model_type = 'smpl'
        ext = 'pkl'
        num_expression_coeffs = 10
        use_face_contour = False

        gender = 'male'
        num_betas = 10
        betas = np.zeros(num_betas)
        root_orient = torch.Tensor(pose[:, :3])
        body_pose = torch.Tensor(pose[:, 3:72])
        betas = torch.Tensor(np.repeat(betas[:num_betas][np.newaxis], repeats=min_num, axis=0))

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
        seq_len = 100
        fs = np.arange(0, num_frames - seq_len + 1)  # 窗口滑动
        fs_sel = fs
        for i in np.arange(seq_len - 1):
            fs_sel = np.vstack((fs_sel, fs + i + 1))  # 沿着竖直方向将矩阵堆叠起来将fs_sel和fs+i+1放到一起
        fs_sel = fs_sel.transpose()  # eg.((35, 1704)变成(1704, 35))
        amass_acc_sam = acc[fs_sel, :].numpy()
        amass_ori_sam = ori[fs_sel, :].numpy()
        amass_poses_sam = pose[fs_sel, :].numpy()
        amass_joints_sam = joints_pos[fs_sel, :]
        # 将数据拼接到一起
        for a in range(amass_acc_sam.shape[0]):
            i += 1
            data_out = {'ori': amass_ori_sam[a], 'acc': amass_acc_sam[a], 'poses': amass_poses_sam[a],
                        'joints': amass_joints_sam[a]}
            # 这里创建数据输出文件夹
            out_dir = os.path.join(out_path, "total" + str(i) + '.pkl')
            with open(out_dir, 'wb') as fout:
                pkl.dump(data_out, fout)


def Sysamass():
    amass_dataset_path = r"/data/xt/dataset/Synthetic_60FPS"
    output_dataset_path = r"/data/xt/dataset/DIP_IMU_AMASS/train"
    comp_device = torch.device("cpu")
    print("comp_device", comp_device)
    subset_dir = [x for x in os.listdir(amass_dataset_path)
                  if os.path.isdir(amass_dataset_path + '/' + x)]
    print(subset_dir)
    i = 0
    for subset in subset_dir:
        subset_dir = os.path.join(amass_dataset_path, subset)
        seqs = glob.glob(os.path.join(subset_dir, '*.pkl'))
        print('-- processing subset {:s}'.format(subset_dir))
        # main loop to process each sequence
        for seq in tqdm(seqs):
            print(seq)
            if os.path.basename(seq) == 'shape.npz':
                continue

            data = pkl.load(open(seq, 'rb'), encoding='latin1')

            acc = torch.tensor(data['acc'])
            ori = torch.tensor(data['ori'])
            poses = torch.tensor(data['poses'])
            print(acc.shape, ori.shape, poses.shape)
            if len(poses) < 300:
                continue
            for _ in range(4):
                acc[1:].masked_scatter_(torch.isnan(acc[1:]), acc[:-1][torch.isnan(acc[1:])])
                ori[1:].masked_scatter_(torch.isnan(ori[1:]), ori[:-1][torch.isnan(ori[1:])])
                acc[:-1].masked_scatter_(torch.isnan(acc[:-1]), acc[1:][torch.isnan(acc[:-1])])
                ori[:-1].masked_scatter_(torch.isnan(ori[:-1]), ori[1:][torch.isnan(ori[:-1])])

            acc, ori, poses = acc[6:-6], ori[6:-6], poses[6:-6]
            n_frames = poses.shape[0]
            smpl_folder = "/data/xt/body_models/vPOSE/models"

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
            poses = poses[:, :72]
            print(joints_pos.shape)
            num_frames = len(joints_pos)
            seq_len = 100
            fs = np.arange(0, num_frames - seq_len + 1)  # 窗口滑动
            fs_sel = fs
            for i in np.arange(seq_len - 1):
                fs_sel = np.vstack((fs_sel, fs + i + 1))  # 沿着竖直方向将矩阵堆叠起来将fs_sel和fs+i+1放到一起
            fs_sel = fs_sel.transpose()  # eg.((35, 1704)变成(1704, 35))
            amass_acc_sam = acc[fs_sel, :].numpy()
            amass_ori_sam = ori[fs_sel, :].numpy()
            amass_poses_sam = poses[fs_sel, :].numpy()
            amass_joints_sam = joints_pos[fs_sel, :]
            # 将数据拼接到一起
            for a in range(amass_acc_sam.shape[0]):
                i += 1
                data_out = {'ori': amass_ori_sam[a], 'acc': amass_acc_sam[a], 'poses': amass_poses_sam[a],
                            'joints': amass_joints_sam[a]}
                # 这里创建数据输出文件夹
                out_dir = os.path.join(output_dataset_path, subset + str(i) + '.pkl')
                with open(out_dir, 'wb') as fout:
                    pkl.dump(data_out, fout)


# read data

# Turn MoCap data into 60FPS
def interpolation_integer(poses_ori, fps):
    poses = []
    n_tmp = int(fps / TARGET_FPS)
    poses_ori = poses_ori[::n_tmp]

    for t in poses_ori:
        poses.append(t)

    return poses


def get_ori_acc(poses, joints_pos, orientation, vertex, frame_rate):
    acceleration = []
    n = 4
    time_interval = n * (1.0 / frame_rate)  # 参考SIGGRAPH的论文的数据设定。
    total_number = vertex.shape[0]
    for idx in range(n, total_number - n):
        vertex_0 = vertex[idx - n].astype(float)  # 6*3
        vertex_1 = vertex[idx].astype(float)
        vertex_2 = vertex[idx + n].astype(float)
        accel_tmp = (vertex_2 + vertex_0 - 2 * vertex_1) / (time_interval * time_interval)
        acceleration.append(accel_tmp)

    return poses[n:-n], joints_pos[n:-n], orientation[n:-n], acceleration


if __name__ == "__main__":
    option = Options().parse()
    Sys_amass(option)
    # Sys_dip(option)
    # Sys_dip_path()
    # Systotal()
