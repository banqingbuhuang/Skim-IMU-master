from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os

import pickle

import numpy as np
import smplx
import torch
import pickle as pkl
from tqdm import tqdm

# TODO
TARGET_FPS = 60

acc_scale = 30


def Systotal(total_path, out_path, smpl_folder):
    comp_device = torch.device("cpu")
    print("comp_device", comp_device)
    # subset_dir = [x for x in os.listdir(total_path)
    #               if os.path.isdir(total_path + '/' + x)]
    # for subset in subset_dir:
    #     print(subset)
    seqs = glob.glob(os.path.join(total_path, '*.pkl'))
    idx = 0
    for seq in tqdm(seqs):
        print(seq)
        data = pkl.load(open(seq, 'rb'), encoding='latin1')
        ori = torch.from_numpy(data['ori']).float()
        acc = torch.from_numpy(data['acc']).float()
        pose = torch.from_numpy(data['gt']).float()
        print(acc.shape, ori.shape, pose.shape)
        min_num = min(acc.shape[0], ori.shape[0], pose.shape[0])
        acc = acc[:min_num]
        ori = ori[:min_num]
        pose = pose[:min_num]
        smpl_folder = smpl_folder

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
        seq_len = 60
        batch_num = num_frames // seq_len
        all_batch_num = batch_num * seq_len
        amass_acc_sam = acc[:all_batch_num, :].view(batch_num, seq_len, 6, 3)
        amass_ori_sam = ori[:all_batch_num, :].view(batch_num, seq_len, 6, 3, 3)
        amass_poses_sam = pose[:all_batch_num, :].view(batch_num, seq_len, 24, 3)
        amass_joints_sam = joints_pos[:all_batch_num, :].reshape(batch_num, seq_len, 24, 3)
        # fs = np.arange(0, num_frames - seq_len + 1)  # 窗口滑动
        # fs_sel = fs
        # for i in np.arange(seq_len - 1):
        #     fs_sel = np.vstack((fs_sel, fs + i + 1))  # 沿着竖直方向将矩阵堆叠起来将fs_sel和fs+i+1放到一起
        # fs_sel = fs_sel.transpose()  # eg.((35, 1704)变成(1704, 35))
        # amass_acc_sam = acc[fs_sel, :].numpy()
        # amass_ori_sam = ori[fs_sel, :].numpy()
        # amass_poses_sam = pose[fs_sel, :].numpy()
        # amass_joints_sam = joints_pos[fs_sel, :]
        # 将数据拼接到一起
        for a in range(amass_acc_sam.shape[0]):
            idx += 1
            data_out = {'ori': amass_ori_sam[a], 'acc': amass_acc_sam[a], 'poses': amass_poses_sam[a],
                        'joints': amass_joints_sam[a]}
            # 这里创建数据输出文件夹
            out_dir = os.path.join(out_path, "total" + str(idx) + '.pkl')
            with open(out_dir, 'wb') as fout:
                pkl.dump(data_out, fout)
        print(idx)


def loadDIP_nn(path, out_path, smpl_folder):
    # 0:head,2 ,belly 7:lelbow,13:lwrist,11:lknee,15:lankel,8:relbow,14:rwrist,12:rknee,16:rankle
    imu_mask = [7, 8, 11, 12, 0, 2]
    # imu_mask = [2, 0, 7, 8, 12, 14, 11, 12, 15, 16]
    subset_dir = [x for x in os.listdir(path)
                  if os.path.isdir(path + '/' + x)]
    for subset in subset_dir:
        print(subset)
        idx = 0
        # 数据输入
        # 数据的拼接
        seqs = glob.glob(os.path.join(path, subset, '*.pkl'))
        print('-- processing subset {:s}'.format(subset))
        for seq in tqdm(seqs):
            print(seq)
            data = pickle.load(open(seq, 'rb'), encoding='latin1')
            acc = torch.from_numpy(data['imu_acc'][:, imu_mask]).float()  # n,3
            ori = torch.from_numpy(data['imu_ori'][:, imu_mask]).float()  # n,3,3
            poses = torch.from_numpy(data['gt']).float()  # n,72
            # fill nan with nearest neighbors
            for _ in range(4):
                acc[1:].masked_scatter_(torch.isnan(acc[1:]), acc[:-1][torch.isnan(acc[1:])])
                ori[1:].masked_scatter_(torch.isnan(ori[1:]), ori[:-1][torch.isnan(ori[1:])])
                acc[:-1].masked_scatter_(torch.isnan(acc[:-1]), acc[1:][torch.isnan(acc[:-1])])
                ori[:-1].masked_scatter_(torch.isnan(ori[:-1]), ori[1:][torch.isnan(ori[:-1])])
            acc, ori, poses = acc[6:-6], ori[6:-6], poses[6:-6]
            if torch.isnan(acc).sum() == 0 and torch.isnan(ori).sum() == 0 and torch.isnan(poses).sum() == 0:
                mocap_framerate = 60
                if mocap_framerate % TARGET_FPS == 0:
                    n_tmp = int(mocap_framerate / TARGET_FPS)
                    acc = acc[::n_tmp].numpy()
                    ori = ori[::n_tmp].numpy()
                    poses = poses[::n_tmp].numpy()
                n_frames = poses.shape[0]
                smpl_folder = smpl_folder
                # smpl_folder = "C:/Gtrans/body_models/vPOSE/models"
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
                seq_len = 60
                batch_num = num_frames // seq_len
                all_batch_num = batch_num * seq_len
                amass_acc_sam = acc[:all_batch_num, :].reshape(batch_num, seq_len, 6, 3)
                amass_ori_sam = ori[:all_batch_num, :].reshape(batch_num, seq_len, 6, 3, 3)
                amass_poses_sam = poses[:all_batch_num, :].reshape(batch_num, seq_len, 24, 3)
                amass_joints_sam = joints_pos[:all_batch_num, :].reshape(batch_num, seq_len, 24, 3)
                # fs = np.arange(0, num_frames - seq_len + 1)  # 窗口滑动
                # fs_sel = fs
                # for i in np.arange(seq_len - 1):
                #     fs_sel = np.vstack((fs_sel, fs + i + 1))  # 沿着竖直方向将矩阵堆叠起来将fs_sel和fs+i+1放到一起
                # fs_sel = fs_sel.transpose()  # eg.((35, 1704)变成(1704, 35))
                # amass_acc_sam = acc[fs_sel, :]
                # amass_ori_sam = ori[fs_sel, :]
                # amass_poses_sam = poses[fs_sel, :]
                # amass_joints_sam = joints_pos[fs_sel, :]
                # 将数据拼接到一起
                for a in range(amass_acc_sam.shape[0]):
                    idx += 1
                    data_out = {'ori': amass_ori_sam[a], 'acc': amass_acc_sam[a], 'poses': amass_poses_sam[a],
                                'joints': amass_joints_sam[a]}
                    # 这里创建数据输出文件夹
                    out_dir = os.path.join(out_path, subset + str(idx) + '.pkl')
                    with open(out_dir, 'wb') as fout:
                        pkl.dump(data_out, fout)
                print(idx)
            else:
                print('DIP-IMU: %s has too much nan! Discard!' % seq)


def loadDIP(path, out_path, smpl_folder):
    # 0:head,2 ,belly 7:lelbow,13:lwrist,11:lknee,15:lankel,8:relbow,14:rwrist,12:rknee,16:rankle
    imu_mask = [7, 8, 11, 12, 0, 2]
    # imu_mask = [2, 0, 7, 8, 12, 14, 11, 12, 15, 16]
    idx = 0
    seqs = glob.glob(os.path.join(path, '*.pkl'))
    idx = 0
    for seq in tqdm(seqs):
        print(seq)
        data = pickle.load(open(seq, 'rb'), encoding='latin1')
        acc = torch.from_numpy(data['imu_acc'][:, imu_mask]).float()  # n,3
        ori = torch.from_numpy(data['imu_ori'][:, imu_mask]).float()  # n,3,3
        poses = torch.from_numpy(data['gt']).float()  # n,72
        # fill nan with nearest neighbors
        for _ in range(4):
            acc[1:].masked_scatter_(torch.isnan(acc[1:]), acc[:-1][torch.isnan(acc[1:])])
            ori[1:].masked_scatter_(torch.isnan(ori[1:]), ori[:-1][torch.isnan(ori[1:])])
            acc[:-1].masked_scatter_(torch.isnan(acc[:-1]), acc[1:][torch.isnan(acc[:-1])])
            ori[:-1].masked_scatter_(torch.isnan(ori[:-1]), ori[1:][torch.isnan(ori[:-1])])
        acc, ori, poses = acc[6:-6], ori[6:-6], poses[6:-6]
        if torch.isnan(acc).sum() == 0 and torch.isnan(ori).sum() == 0 and torch.isnan(poses).sum() == 0:
            mocap_framerate = 60
            if mocap_framerate % TARGET_FPS == 0:
                n_tmp = int(mocap_framerate / TARGET_FPS)
                acc = acc[::n_tmp].numpy()
                ori = ori[::n_tmp].numpy()
                poses = poses[::n_tmp].numpy()
            n_frames = poses.shape[0]
            smpl_folder = smpl_folder
            # smpl_folder = "C:/Gtrans/body_models/vPOSE/models"
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
            seq_len = 60
            # batch_num = num_frames // seq_len
            # all_batch_num = batch_num * seq_len
            # amass_acc_sam = acc[:all_batch_num, :].reshape(batch_num, seq_len, 6, 3)
            # amass_ori_sam = ori[:all_batch_num, :].reshape(batch_num, seq_len, 6, 3, 3)
            # amass_poses_sam = poses[:all_batch_num, :].reshape(batch_num, seq_len, 24, 3)
            # amass_joints_sam = joints_pos[:all_batch_num, :].reshape(batch_num, seq_len, 24, 3)
            fs = np.arange(0, num_frames - seq_len + 1)  # 窗口滑动
            fs_sel = fs
            for i in np.arange(seq_len - 1):
                fs_sel = np.vstack((fs_sel, fs + i + 1))  # 沿着竖直方向将矩阵堆叠起来将fs_sel和fs+i+1放到一起
            fs_sel = fs_sel.transpose()  # eg.((35, 1704)变成(1704, 35))
            amass_acc_sam = acc[fs_sel, :]
            amass_ori_sam = ori[fs_sel, :]
            amass_poses_sam = poses[fs_sel, :]
            amass_joints_sam = joints_pos[fs_sel, :]
            # 将数据拼接到一起
            for a in range(amass_acc_sam.shape[0]):
                idx += 1
                data_out = {'ori': amass_ori_sam[a], 'acc': amass_acc_sam[a], 'poses': amass_poses_sam[a],
                            'joints': amass_joints_sam[a]}
                # 这里创建数据输出文件夹
                out_dir = os.path.join(out_path, 'inter' + str(idx) + '.pkl')
                with open(out_dir, 'wb') as fout:
                    pkl.dump(data_out, fout)
            print(idx)
        else:
            print('DIP-IMU: %s has too much nan! Discard!' % seq)

if __name__ == '__main__':
    # def load_amass_data():
    # DIP_dataset_path = "/data/xt/dataset/dip/test"
    # output_dataset_path = "/data/xt/dataset/DIP-total_2/pose_test"
    # smpl_path = "/data/xt/body_models/vPOSE/models"
    # # loadDIP_nn(DIP_dataset_path, output_dataset_path, smpl_path)
    # total_path = "/data/xt/dataset/totalCapture/test/walking"
    # total_out_path = "/data/xt/dataset/DIP-total_2/total_pose/walking"
    # Systotal(total_path, total_out_path, smpl_path)
    DIP_dataset_path = "C:/Gtrans/dataset/IMU Dataset/DIP-test/05"
    output_dataset_path = "C:/Gtrans/dataset/dataset/dip_test/05"
    smpl_path = "C:/Gtrans/body_models/vPOSE/models"
    loadDIP(DIP_dataset_path, output_dataset_path, smpl_path)
