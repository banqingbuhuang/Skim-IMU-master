import torch.nn.functional as F
import numpy as np
import smplx
# Smpl的关节位置与目前是个关节位置的顺序对应
# 0:根节点，15头，23左手腕，8左脚踝，22右手腕，7右脚踝，19：左手肘，18右手肘，5左膝盖，4右膝盖
# Joint_idx = [0, 15, 23, 22, 5, 4, 19, 18, 8, 7]
from utils.utils_math.angular import RotationRepresentation

from utils.SMPLmodel import ParametricModel
import torch
from utils.utils_math import angular as A

Joint_idx = [0, 15, 23, 22, 5, 4, 19, 18, 8, 7]


# 0: 根节点，1：头，2：左手腕，3：右手腕，4：左膝盖，5:右膝盖，6：左手肘，7：右手肘，8：左脚踝，9：右脚踝

# 0：根节点，1:头，2:左手肘，3:右手肘，4:左膝盖，5:右膝盖，6:左手腕，7:右手腕，8左脚踝，9:右脚踝

def bone_loss(y_out, out_joints):
    batch, frame, node, dim = out_joints.shape
    # b1_out计算左手腕到左手肘的长度2-6:b2_out计算右手腕到右手肘的长度3-7:b3_out计算左膝盖到左脚踝的长度4-8:b4_out计算右膝盖到右脚踝的长度5-9
    loss = length(y_out, out_joints)
    return loss


# y_out:torch.Size([32, 24, 3, 60]
# out_joints:torch.Size([32, 24, 3, 60]
def position_loss_gai(y_out, out_joints, loss_weight):
    err = torch.norm(y_out - out_joints, 2, dim=3)
    weight_err = torch.mean(torch.mul(err, loss_weight))
    return weight_err


def position_loss_ignore(y_out, out_joints, ignore):
    y_out[:, :, ignore] = torch.ones(3, device=y_out.device)
    out_joints[:, :, ignore] = torch.ones(3, device=y_out.device)
    # 这么改是会将原来的也一起改掉的，所以需要先计算那个position_loss
    y_out = y_out.view(-1, 3)
    out_joints = out_joints.view(-1, 3)
    mean_3d_err = torch.mean(torch.norm(y_out - out_joints, 2, dim=1))
    return mean_3d_err


def position_loss_yi(y_out, out_joints, ignore):
    y_out[:, ignore] = torch.ones(3, device=y_out.device)
    out_joints[:, ignore] = torch.ones(3, device=y_out.device)
    offset = (out_joints[:, 0] - y_out[:, 0]).unsqueeze(1)
    mean_3d_err = (y_out + offset - out_joints).norm(dim=2)
    return mean_3d_err.mean()


def position_loss(y_out, out_joints):
    y_out = y_out.reshape(-1, 3)
    out_joints = out_joints.reshape(-1, 3)
    mean_3d_err = torch.mean(torch.norm(y_out - out_joints, 2, 1))
    return mean_3d_err


def bone(n_10, out_joints, all_n):
    batch, frame, node, dim = out_joints.shape
    out_joints = torch.cat([out_joints[:, :, idx, :] for idx in Joint_idx], dim=2).reshape(batch, frame, dim, -1)
    out_joints = torch.transpose(out_joints, 2, 3)
    n_10 = n_10.reshape(-1, all_n, 10, 3)
    j2 = n_10[:, :, 2, :].reshape(-1, 3).contiguous()
    j6 = n_10[:, :, 6, :].reshape(-1, 3).contiguous()
    len26 = torch.norm(j2 - j6, 2, 1)

    j2_t = out_joints[:, :, 2, :].reshape(-1, 3).contiguous()
    j6_t = out_joints[:, :, 6, :].reshape(-1, 3).contiguous()
    len26_t = torch.norm(j2_t - j6_t, 2, 1)
    loss_2_6 = torch.mean(torch.abs(len26 - len26_t))

    j4 = n_10[:, :, 4, :].reshape(-1, 3).contiguous()
    j7 = n_10[:, :, 7, :].reshape(-1, 3).contiguous()
    len47 = torch.norm(j4 - j7, 2, 1)
    j4_t = out_joints[:, :, 4, :].reshape(-1, 3).contiguous()
    j7_t = out_joints[:, :, 7, :].reshape(-1, 3).contiguous()
    len47_t = torch.norm(j4_t - j7_t, 2, 1)  # (x1-x2)^2,(y1-y2)^2,(z1-z2)^2
    loss_3_7 = torch.mean(torch.abs(len47 - len47_t))

    j3 = n_10[:, :, 3, :].reshape(-1, 3).contiguous()
    j8 = n_10[:, :, 8, :].reshape(-1, 3).contiguous()
    len38 = torch.norm(j3 - j8, 2, 1)  # (x1-x2)^2,(y1-y2)^2,(z1-z2)^2
    j3_t = out_joints[:, :, 3, :].reshape(-1, 3).contiguous()
    j8_t = out_joints[:, :, 8, :].reshape(-1, 3).contiguous()
    len38_t = torch.norm(j4_t - j8_t, 2, 1)  # (x1-x2)^2,(y1-y2)^2,(z1-z2)^2
    loss_4_8 = torch.mean(torch.abs(len38 - len38_t))

    j5 = n_10[:, :, 5, :].reshape(-1, 3).contiguous()
    j9 = n_10[:, :, 9, :].reshape(-1, 3).contiguous()
    len59 = torch.norm(j5 - j9, 2, 1)  # (x1-x2)^2,(y1-y2)^2,(z1-z2)^2
    j5_t = out_joints[:, :, 5, :].reshape(-1, 3).contiguous()
    j9_t = out_joints[:, :, 9, :].reshape(-1, 3).contiguous()
    len59_t = torch.norm(j5_t - j9_t, 2, 1)  # (x1-x2)^2,(y1-y2)^2,(z1-z2)^2
    loss_5_9 = torch.mean(torch.abs(len59 - len59_t))

    all_loss = loss_2_6 + loss_3_7 + loss_4_8 + loss_5_9
    return all_loss


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


def angle_error(pose_p, pose_t):
    err = A.radian_to_degree(A.angle_between(pose_p, pose_t).view(pose_p.shape[0], -1))
    return err.mean()


def poses_loss_degree(y_out, out_poses):
    gae = A.radian_to_degree(A.angle_between(y_out, out_poses).view(out_poses.shape[0], -1))  # N, J
    return torch.mean(gae)


def poses_loss_yi(y_out, out_poses):
    batch, frame, node, dim = y_out.shape
    pose_p = y_out.clone().view(-1, 24, 3, 3)
    pose_t = out_poses.clone().view(-1, 24, 3, 3)
    pose_p[:, joint_set.ignored] = torch.eye(3, device=pose_p.device)
    pose_t[:, joint_set.ignored] = torch.eye(3, device=pose_t.device)
    pose_p = pose_p.reshape(-1, dim)
    pose_t = pose_t.reshape(-1, dim)
    loss = torch.mean(torch.norm(pose_p - pose_t, 2, 1))
    return loss


def poses_loss_gai(y_out, out_poses, loss_weight):
    batch, frame, node, dim = y_out.shape
    out_poses = out_poses.reshape(batch, frame, node, -1)
    err = torch.norm(y_out - out_poses, 2, dim=3)
    weight_err = torch.mean(torch.mul(err, loss_weight))
    return weight_err


def poses_loss(y_out, out_poses):
    y_out = y_out.reshape(-1, 9)
    out_poses = out_poses.reshape(-1, 9)
    loss = torch.mean(torch.norm(y_out - out_poses, 2, 1))
    return loss


# 0:根节点，15头，23左手腕，8左脚踝，22右手腕，7右脚踝，19：左手肘，18右手肘，5左膝盖，4右膝盖
# 23-19,22-28,7-4,8-5
def length(n_10, out_joints):
    j23 = n_10[:, :, 23, :].reshape(-1, 3).contiguous()
    j19 = n_10[:, :, 19, :].reshape(-1, 3).contiguous()
    len26 = F.pairwise_distance(j23, j19, p=2)

    j23_t = out_joints[:, :, 23, :].reshape(-1, 3).contiguous()
    j19_t = out_joints[:, :, 19, :].reshape(-1, 3).contiguous()
    len26_t = F.pairwise_distance(j23_t, j19_t, p=2)
    loss_2_6 = torch.mean(torch.abs(len26 - len26_t))
    j22 = n_10[:, :, 22, :].reshape(-1, 3).contiguous()
    j18 = n_10[:, :, 18, :].reshape(-1, 3).contiguous()
    len37 = F.pairwise_distance(j22, j18, p=2)
    j22_t = out_joints[:, :, 22, :].reshape(-1, 3).contiguous()
    j18_t = out_joints[:, :, 18, :].reshape(-1, 3).contiguous()
    len37_t = F.pairwise_distance(j22_t, j18_t, p=2)  # (x1-x2)^2,(y1-y2)^2,(z1-z2)^2
    loss_3_7 = torch.mean(torch.abs(len37 - len37_t))

    j4 = n_10[:, :, 4, :].reshape(-1, 3).contiguous()
    j7 = n_10[:, :, 7, :].reshape(-1, 3).contiguous()
    len47 = F.pairwise_distance(j4, j7, p=2)  # (x1-x2)^2,(y1-y2)^2,(z1-z2)^2
    j4_t = out_joints[:, :, 4, :].reshape(-1, 3).contiguous()
    j7_t = out_joints[:, :, 7, :].reshape(-1, 3).contiguous()
    len47_t = F.pairwise_distance(j4_t, j7_t, p=2)  # 求均值
    loss_4_7 = torch.mean(torch.abs(len47 - len47_t))

    j5 = n_10[:, :, 5, :].reshape(-1, 3).contiguous()
    j8 = n_10[:, :, 8, :].reshape(-1, 3).contiguous()
    len58 = F.pairwise_distance(j5, j8, p=2)  # (x1-x2)^2,(y1-y2)^2,(z1-z2)^2
    j5_t = out_joints[:, :, 5, :].reshape(-1, 3).contiguous()
    j8_t = out_joints[:, :, 8, :].reshape(-1, 3).contiguous()
    len58_t = F.pairwise_distance(j5_t, j8_t, p=2)  # (x1-x2)^2,(y1-y2)^2,(z1-z2)^2
    loss_5_8 = torch.mean(torch.abs(len58 - len58_t))

    all_loss = loss_2_6 + loss_3_7 + loss_4_7 + loss_5_8
    return all_loss / 4


def frame_length(test_out, test_joints):
    batch, _ = test_joints.shape
    test_out = test_out.reshape(batch, 24, 3)
    test_joints = test_joints.reshape(batch, 24, 3)
    j23 = test_out[:, 23, :].reshape(-1, 3).contiguous()
    j19 = test_out[:, 19, :].reshape(-1, 3).contiguous()
    len26 = F.pairwise_distance(j23, j19, p=2)

    j23_t = test_joints[:, 23, :].reshape(-1, 3).contiguous()
    j19_t = test_joints[:, 19, :].reshape(-1, 3).contiguous()
    len26_t = torch.mean(F.pairwise_distance(j23_t, j19_t, p=2))
    loss_2_6 = torch.mean(torch.abs(len26 - len26_t))
    j22 = test_out[:, 22, :].reshape(-1, 3).contiguous()
    j18 = test_out[:, 18, :].reshape(-1, 3).contiguous()
    len37 = F.pairwise_distance(j22, j18, p=2)
    j22_t = test_joints[:, 22, :].reshape(-1, 3).contiguous()
    j18_t = test_joints[:, 18, :].reshape(-1, 3).contiguous()
    len37_t = torch.mean(F.pairwise_distance(j22_t, j18_t, p=2))  # (x1-x2)^2,(y1-y2)^2,(z1-z2)^2
    loss_3_7 = torch.mean(torch.abs(len37 - len37_t))

    j4 = test_out[:, 4, :].reshape(-1, 3).contiguous()
    j7 = test_out[:, 7, :].reshape(-1, 3).contiguous()
    len47 = F.pairwise_distance(j4, j7, p=2)  # (x1-x2)^2,(y1-y2)^2,(z1-z2)^2
    j4_t = test_joints[:, 4, :].reshape(-1, 3).contiguous()
    j7_t = test_joints[:, 7, :].reshape(-1, 3).contiguous()
    len47_t = torch.mean(F.pairwise_distance(j4_t, j7_t, p=2))  # 求均值
    loss_4_7 = torch.mean(torch.abs(len47 - len47_t))

    j5 = test_out[:, 5, :].reshape(-1, 3).contiguous()
    j8 = test_out[:, 8, :].reshape(-1, 3).contiguous()
    len58 = F.pairwise_distance(j5, j8, p=2)  # (x1-x2)^2,(y1-y2)^2,(z1-z2)^2
    j5_t = test_joints[:, 5, :].reshape(-1, 3).contiguous()
    j8_t = test_joints[:, 8, :].reshape(-1, 3).contiguous()
    len58_t = torch.mean(F.pairwise_distance(j5_t, j8_t, p=2))  # (x1-x2)^2,(y1-y2)^2,(z1-z2)^2
    loss_5_8 = torch.mean(torch.abs(len58 - len58_t))

    all_loss = loss_2_6 + loss_3_7 + loss_4_7 + loss_5_8
    return all_loss


class BasePoseEvaluator:
    r"""
    Base class for evaluators that evaluate motions.
    """

    def __init__(self, official_model_file: str, rep=RotationRepresentation.ROTATION_MATRIX, use_pose_blendshape=False,
                 device=torch.device('cpu')):
        self.model = ParametricModel(official_model_file, use_pose_blendshape=use_pose_blendshape, device=device)
        self.rep = rep
        self.device = device

    def _preprocess(self, pose, shape=None, tran=None):
        pose = A.to_rotation_matrix(pose.to(self.device), self.rep).view(pose.shape[0], -1)
        shape = shape.to(self.device) if shape is not None else shape
        tran = tran.to(self.device) if tran is not None else tran
        return pose, shape, tran


class FullMotionEvaluator(BasePoseEvaluator):
    r"""
    Evaluator for full motions (pose sequences with global translations). Plenty of metrics.
    """

    def __init__(self, official_model_file: str, align_joint=None, rep=RotationRepresentation.ROTATION_MATRIX,
                 use_pose_blendshape=False, fps=60, joint_mask=None, device=torch.device('cpu')):
        r"""
        Init a full motion evaluator.

        :param official_model_file: Path to the official SMPL/MANO/SMPLH model to be loaded.
        :param align_joint: Which joint to align. (e.g. SMPLJoint.ROOT). By default the root.
        :param rep: The rotation representation used in the input poses.
        :param use_pose_blendshape: Whether to use pose blendshape or not.
        :param joint_mask: If not None, local angle error, global angle error, and joint position error
                           for these joints will be calculated additionally.
        :param fps: Motion fps, by default 60.
        :param device: torch.device, cpu or cuda.
        """
        super(FullMotionEvaluator, self).__init__(official_model_file, rep, use_pose_blendshape, device=device)
        self.align_joint = 0 if align_joint is None else align_joint.value
        self.fps = fps
        self.joint_mask = joint_mask

    def __call__(self, pose_p, pose_t, shape_p=None, shape_t=None, tran_p=None, tran_t=None):
        r"""
        Get the measured errors. The returned tensor in shape [10, 2] contains mean and std of:
          0.  Joint position error (align_joint position aligned).
          1.  Vertex position error (align_joint position aligned).
          2.  Joint local angle error (in degrees).
          3.  Joint global angle error (in degrees).
          4.  Predicted motion jerk (with global translation).
          5.  True motion jerk (with global translation).
          6.  Translation error (mean root translation error per second, using a time window size of 1s).
          7.  Masked joint position error (align_joint position aligned, zero if mask is None).
          8.  Masked joint local angle error. (in degrees, zero if mask is None).
          9.  Masked joint global angle error. (in degrees, zero if mask is None).

        :param pose_p: Predicted pose or the first pose in shape [batch_size, *] that can
                       reshape to [batch_size, num_joint, rep_dim].
        :param pose_t: True pose or the second pose in shape [batch_size, *] that can
                       reshape to [batch_size, num_joint, rep_dim].
        :param shape_p: Predicted shape that can expand to [batch_size, 10]. Use None for the mean(zero) shape.
        :param shape_t: True shape that can expand to [batch_size, 10]. Use None for the mean(zero) shape.
        :param tran_p: Predicted translations in shape [batch_size, 3]. Use None for zeros.
        :param tran_t: True translations in shape [batch_size, 3]. Use None for zeros.
        :return: Tensor in shape [10, 2] for the mean and std of all errors.
        """
        f = self.fps
        pose_local_p, shape_p, tran_p = self._preprocess(pose_p, shape_p, tran_p)
        pose_local_t, shape_t, tran_t = self._preprocess(pose_t, shape_t, tran_t)
        pose_global_p, joint_p, vertex_p = self.model.forward_kinematics(pose_local_p, shape_p, tran_p, calc_mesh=True)
        pose_global_t, joint_t, vertex_t = self.model.forward_kinematics(pose_local_t, shape_t, tran_t, calc_mesh=True)

        offset_from_p_to_t = (joint_t[:, self.align_joint] - joint_p[:, self.align_joint]).unsqueeze(1)
        ve = (vertex_p + offset_from_p_to_t - vertex_t).norm(dim=2)  # N, J
        je = (joint_p + offset_from_p_to_t - joint_t).norm(dim=2)  # N, J

        gae = A.radian_to_degree(A.angle_between(pose_global_p, pose_global_t).view(pose_p.shape[0], -1))  # N, J
        lae = A.radian_to_degree(A.angle_between(pose_local_p, pose_local_t).view(pose_p.shape[0], -1))  # N, J
        mgae = gae[:, self.joint_mask] if self.joint_mask is not None else torch.zeros(1)  # N, mJ
        out = torch.tensor([torch.mean(mgae), torch.mean(gae), torch.mean(lae), torch.mean(je), torch.mean(ve)]
                           )
        print(out)
        return out


def posetomesh(pose, smpl_folder="/data/wwu/xt/body_models/vPOSE/models"):
    # smpl_folder = "C:/Gtrans/body_models/vPOSE/models"
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
    joint = joints[:, :24].contiguous()
    vertices = output.vertices.detach()
    return poses, joint, vertices


class allJointMotionEvaluator:
    r"""
    Evaluator for full motions (pose sequences with global translations). Plenty of metrics.
    """

    def __init__(self, smpl_folder: str, align_joint=None, rep=RotationRepresentation.ROTATION_MATRIX,
                 use_pose_blendshape=False, fps=60, joint_mask=None, device=torch.device('cpu')):
        r"""
        Init a full motion evaluator.

        :param official_model_file: Path to the official SMPL/MANO/SMPLH model to be loaded.
        :param align_joint: Which joint to align. (e.g. SMPLJoint.ROOT). By default the root.
        :param rep: The rotation representation used in the input poses.
        :param use_pose_blendshape: Whether to use pose blendshape or not.
        :param joint_mask: If not None, local angle error, global angle error, and joint position error
                           for these joints will be calculated additionally.
        :param fps: Motion fps, by default 60.
        :param device: torch.device, cpu or cuda.
        """
        super(allJointMotionEvaluator, self).__init__()
        self.align_joint = 0 if align_joint is None else align_joint.value
        self.fps = fps
        self.joint_mask = joint_mask
        self.smpl_folder = smpl_folder

    def __call__(self, pose_p, pose_t, shape_p=None, shape_t=None, tran_p=None, tran_t=None):
        r"""
        Get the measured errors. The returned tensor in shape [10, 2] contains mean and std of:
          0.  Joint position error (align_joint position aligned).
          1.  Vertex position error (align_joint position aligned).
          2.  Joint local angle error (in degrees).
          3.  Joint global angle error (in degrees).
          4.  Predicted motion jerk (with global translation).
          5.  True motion jerk (with global translation).
          6.  Translation error (mean root translation error per second, using a time window size of 1s).
          7.  Masked joint position error (align_joint position aligned, zero if mask is None).
          8.  Masked joint local angle error. (in degrees, zero if mask is None).
          9.  Masked joint global angle error. (in degrees, zero if mask is None).

        :param pose_p: Predicted pose or the first pose in shape [batch_size, *] that can
                       reshape to [batch_size, num_joint, rep_dim].
        :param pose_t: True pose or the second pose in shape [batch_size, *] that can
                       reshape to [batch_size, num_joint, rep_dim].
        :param shape_p: Predicted shape that can expand to [batch_size, 10]. Use None for the mean(zero) shape.
        :param shape_t: True shape that can expand to [batch_size, 10]. Use None for the mean(zero) shape.
        :param tran_p: Predicted translations in shape [batch_size, 3]. Use None for zeros.
        :param tran_t: True translations in shape [batch_size, 3]. Use None for zeros.
        :return: Tensor in shape [10, 2] for the mean and std of all errors.
        """
        batch, node = pose_p.shape[0], pose_p.shape[1]

        gae = A.radian_to_degree(A.angle_between(pose_p, pose_t).view(batch, -1))  # N, J
        mgae = gae[:, self.joint_mask] if self.joint_mask is not None else torch.zeros(1)  # N, mJ
        pose_p = A.rotation_matrix_to_axis_angle(pose_p).view(batch, node, 3).view(batch, -1)
        pose_t = A.rotation_matrix_to_axis_angle(pose_t).view(batch, node, 3).view(batch, -1)
        pose_2_p, joint_p, vertex_p = posetomesh(pose_p, smpl_folder=self.smpl_folder)
        pose_2_t, joint_t, vertex_t = posetomesh(pose_t, smpl_folder=self.smpl_folder)
        gae2 = A.radian_to_degree(
            A.angle_between(pose_2_p, pose_2_t, RotationRepresentation.AXIS_ANGLE).view(batch, -1))  # N, J
        ve_2 = (vertex_p - vertex_t).norm(dim=2)  # N, J
        je_2 = (joint_p - joint_t).norm(dim=2)  # N, J
        out = torch.tensor([torch.mean(mgae),
                            torch.mean(gae),
                            torch.mean(je_2),
                            torch.mean(ve_2),
                            ])
        return out


class PoseEvaluator:
    def __init__(self, official_model_file="/data/wwu/xt/body_models/vPOSE/models/smpl/SMPL_MALE.pkl",
                 smpl_folder="/data/wwu/xt/body_models/vPOSE/models"):
        self._eval_fn = FullMotionEvaluator(
            official_model_file=official_model_file,
            joint_mask=torch.tensor([1, 2, 16, 17]))
        self._eval_aj = allJointMotionEvaluator(
            smpl_folder=smpl_folder,
            joint_mask=torch.tensor([1, 2, 16, 17]))

    def eval(self, pose_p, pose_t):
        pose_p = pose_p.clone().view(-1, 24, 3, 3)
        pose_t = pose_t.clone().view(-1, 24, 3, 3)
        pose_p[:, joint_set.ignored] = torch.eye(3, device=pose_p.device)
        pose_t[:, joint_set.ignored] = torch.eye(3, device=pose_t.device)
        errs = self._eval_fn(pose_p, pose_t)
        return torch.stack([errs[0], errs[1], errs[2], errs[3] * 100, errs[4] * 100])

    def eval_all(self, pose_p, pose_t):
        pose_p = pose_p.clone().view(-1, 24, 3, 3)
        pose_t = pose_t.clone().view(-1, 24, 3, 3)
        # pose_p[:, joint_set.ignored] = torch.eye(3, device=pose_p.device)
        # pose_t[:, joint_set.ignored] = torch.eye(3, device=pose_t.device)
        errs = self._eval_aj(pose_p, pose_t)
        return torch.stack([errs[1], errs[3] * 100])

    @staticmethod
    def print(errors):
        for i, name in enumerate(['SIP Error (deg)', 'Angular Error (deg)', 'Positional Error (cm)',
                                  'Mesh Error (cm)', 'Jitter Error (100m/s^3)']):
            print('%s: %.2f (+/- %.2f)' % (name, errors[i, 0], errors[i, 1]))


class joint_set:
    leaf = [7, 8, 12, 20, 21]
    full = list(range(1, 24))
    reduced = [1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]
    ignored = [0, 7, 8, 10, 11, 20, 21, 22, 23]  # 根节点，手部关节，和脚部关节

    lower_body = [0, 1, 2, 4, 5, 7, 8, 10, 11]
    lower_body_parent = [None, 0, 0, 1, 2, 3, 4, 5, 6]

    n_leaf = len(leaf)
    n_full = len(full)
    n_reduced = len(reduced)
    n_ignored = len(ignored)
