import matplotlib.pyplot as plt
import numpy as np
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
    def __init__(self, ax, lcolor="#3498db", rcolor="#e74c3c"):
        """
        Create a 3d pose visualizer that can be updated with new poses.
        Args
          ax: 3d axis to plot the 3d pose on ：ax = plt.gca(projection='3d')
          lcolor: String. Colour for the left part of the body
          rcolor: String. Colour for the right part of the body
        """

        # Start and endpoints of our representation
        # 表示的起点和终点，表示一条骨骼的始终，23条骨骼
        self.S = np.array([0, 1, 4, 7, 0, 2, 5, 8, 0, 3, 6, 9, 12, 9, 13, 16, 18, 20, 9, 14, 17, 19, 21])
        self.E = np.array([1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12, 15, 13, 16, 18, 20, 22, 14, 17, 19, 21, 23])
        # Left / right indicator
        # 23个，中间的脊骨算左边
        self.ax = ax

        vals = np.zeros((24, 3))

        # Make connection matrix
        # 真值用虚线表示
        self.plots = []
        for i in np.arange(len(self.S)):
            x = np.array([vals[self.S[i], 0], vals[self.E[i], 0]])
            y = np.array([vals[self.S[i], 1], vals[self.E[i], 1]])
            z = np.array([vals[self.S[i], 2], vals[self.E[i], 2]])
            if i == 0:
                self.plots.append(self.ax.plot(x, y, z, lw=1, c=rcolor))
            else:
                self.plots.append(self.ax.plot(x, y, z, lw=1, c=rcolor))
        # 预测的用实线表示
        self.plots_gcndct = []
        for i in np.arange(len(self.S)):
            x = np.array([vals[self.S[i], 0], vals[self.E[i], 0]])
            y = np.array([vals[self.S[i], 1], vals[self.E[i], 1]])
            z = np.array([vals[self.S[i], 2], vals[self.E[i], 2]])
            if i == 0:
                self.plots_gcndct.append(self.ax.plot(x, y, z, lw=1, c=rcolor))
            else:
                self.plots_gcndct.append(self.ax.plot(x, y, z, lw=1, c=rcolor))
        self.plots_residual = []
        for i in np.arange(len(self.S)):
            x = np.array([vals[self.S[i], 0], vals[self.E[i], 0]])
            y = np.array([vals[self.S[i], 1], vals[self.E[i], 1]])
            z = np.array([vals[self.S[i], 2], vals[self.E[i], 2]])
            if i == 0:
                self.plots_residual.append(self.ax.plot(x, y, z, lw=1, c=rcolor))
            else:
                self.plots_residual.append(self.ax.plot(x, y, z, lw=1, c=rcolor))
        self.plots_jinRNN = []
        for i in np.arange(len(self.S)):
            x = np.array([vals[self.S[i], 0], vals[self.E[i], 0]])
            y = np.array([vals[self.S[i], 1], vals[self.E[i], 1]])
            z = np.array([vals[self.S[i], 2], vals[self.E[i], 2]])
            if i == 0:
                self.plots_jinRNN.append(self.ax.plot(x, y, z, lw=1, c=rcolor))
            else:
                self.plots_jinRNN.append(self.ax.plot(x, y, z, lw=1, c=rcolor))
        self.plots_rnngcn = []
        for i in np.arange(len(self.S)):
            x = np.array([vals[self.S[i], 0], vals[self.E[i], 0]])
            y = np.array([vals[self.S[i], 1], vals[self.E[i], 1]])
            z = np.array([vals[self.S[i], 2], vals[self.E[i], 2]])
            if i == 0:
                self.plots_rnngcn.append(self.ax.plot(x, y, z, lw=1, c=rcolor))
            else:
                self.plots_rnngcn.append(self.ax.plot(x, y, z, lw=1, c=rcolor))
        self.plots_rnnpro = []
        for i in np.arange(len(self.S)):
            x = np.array([vals[self.S[i], 0], vals[self.E[i], 0]])
            y = np.array([vals[self.S[i], 1], vals[self.E[i], 1]])
            z = np.array([vals[self.S[i], 2], vals[self.E[i], 2]])
            if i == 0:
                self.plots_rnnpro.append(self.ax.plot(x, y, z, lw=1, c=rcolor))
            else:
                self.plots_rnnpro.append(self.ax.plot(x, y, z, lw=1, c=rcolor))
        self.plots_IPP = []
        for i in np.arange(len(self.S)):
            x = np.array([vals[self.S[i], 0], vals[self.E[i], 0]])
            y = np.array([vals[self.S[i], 1], vals[self.E[i], 1]])
            z = np.array([vals[self.S[i], 2], vals[self.E[i], 2]])
            if i == 0:
                self.plots_IPP.append(self.ax.plot(x, y, z, lw=1, c=rcolor))
            else:
                self.plots_IPP.append(self.ax.plot(x, y, z, lw=1, c=rcolor))
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")
        self.ax.set_axis_off()
        self.ax.axes.get_xaxis().set_visible(False)
        self.ax.axes.get_yaxis().set_visible(False)
        self.ax.axes.get_zaxis().set_visible(False)
        self.ax.view_init(120, -90)
        # self.ax.set_aspect('equal')

    def update(self, gt_channels, pred_channels):
        """
        Update the plotted 3d pose.
        Args
          channels: 96-dim long np array. The pose to plot.
          lcolor: String. Colour for the left part of the body.
          rcolor: String. Colour for the right part of the body.
        Returns
          Nothing. Simply updates the axis with the new pose.
        """
        # assert gt_channels.shape[0] == 72, "channels should have 96 entries, it has %d instead" % gt_channels.shape[0]
        gt_vals = np.reshape(gt_channels, (24, -1))
        lcolor = "#383838"
        rcolor = "#383838"
        for i in np.arange(len(self.S)):
            # 将数据转化为numpy 格式
            x = np.array([gt_vals[self.S[i], 0], gt_vals[self.E[i], 0]])
            y = np.array([gt_vals[self.S[i], 1], gt_vals[self.E[i], 1]])
            z = np.array([gt_vals[self.S[i], 2], gt_vals[self.E[i], 2]])
            self.plots[i][0].set_xdata(x)
            self.plots[i][0].set_ydata(y)
            self.plots[i][0].set_3d_properties(z)
            self.plots[i][0].set_color(lcolor)
            # self.plots[i][0].set_alpha(0.5)
        # assert pred_channels.shape[0] == 72, "channels should have 72 entries, it has %d instead" %
        # pred_channels.shape[0]
        pred_vals = np.reshape(pred_channels, (24, -1))
        lcolor = "#9b59b6"
        rcolor = "#2ecc71"
        for i in np.arange(len(self.S)):
            # 设置关节位置
            x = np.array([pred_vals[self.S[i], 0], pred_vals[self.E[i], 0]])
            y = np.array([pred_vals[self.S[i], 1], pred_vals[self.E[i], 1]])
            z = np.array([pred_vals[self.S[i], 2], pred_vals[self.E[i], 2]])

            self.plots_IPP[i][0].set_xdata(x)
            self.plots_IPP[i][0].set_ydata(y)
            self.plots_IPP[i][0].set_3d_properties(z)
            self.plots_IPP[i][0].set_color(lcolor)
            # self.plots_pred[i][0].set_alpha(0.7)
        # 设置三个轴的范围
        r = 1
        xroot, yroot, zroot = gt_vals[0, 0], gt_vals[0, 1], gt_vals[0, 2]
        self.ax.set_xlim3d([-r + xroot, r + xroot])
        self.ax.set_zlim3d([-r + zroot, r + zroot])
        self.ax.set_ylim3d([-r + yroot, r + yroot])

    def up_x(self, gt_channels, input_color):
        """
            Update the plotted 3d pose.
            Args
              channels: 96-dim long np array. The pose to plot.
              lcolor: String. Colour for the left part of the body.
              rcolor: String. Colour for the right part of the body.
            Returns
              Nothing. Simply updates the axis with the new pose.
            """
        # assert gt_channels.shape[0] == 72, "channels should have 96 entries, it has %d instead" % gt_channels.shape[0]
        gt_vals = np.reshape(gt_channels, (24, -1))
        # "#8e8e8e"  灰色 "#383838" 黑色
        incolor = "#383838"  # 黑色
        oucolor = "#AE433D"
        # i 表示关节位置
        for i in np.arange(len(self.S)):
            # 将数据转化为numpy 格式
            x = np.array([gt_vals[self.S[i], 0], gt_vals[self.E[i], 0]])
            y = np.array([gt_vals[self.S[i], 1], gt_vals[self.E[i], 1]])
            z = np.array([gt_vals[self.S[i], 2], gt_vals[self.E[i], 2]])
            self.plots_IPP[i][0].set_xdata(x)
            self.plots_IPP[i][0].set_ydata(y)
            self.plots_IPP[i][0].set_3d_properties(z)
            self.plots_IPP[i][0].set_color(incolor if input_color == 1 else oucolor)
            # self.plots[i][0].set_alpha(0.5)
        # 设置三个轴的范围
        r = 1
        xroot, yroot, zroot = gt_vals[0, 0], gt_vals[0, 1], gt_vals[0, 2]
        self.ax.set_xlim3d([-r + xroot, r + xroot])
        self.ax.set_zlim3d([-r + zroot, r + zroot])
        self.ax.set_ylim3d([-r + yroot, r + yroot])

    def update_3(self, gt_channels, pro_channels, gcn_channels, pre):
        # assert gt_channels.shape[0] == 72, "channels should have 96 entries, it has %d instead" % gt_channels.shape[0]
        gt_vals = np.reshape(gt_channels, (24, -1))
        gtcolor = "#B3B8BC"
        pcolor = "#AE433D"
        ocolor = "#354E87"
        for i in np.arange(len(self.S)):
            # 将数据转化为numpy 格式
            x = np.array([gt_vals[self.S[i], 0], gt_vals[self.E[i], 0]])
            y = np.array([gt_vals[self.S[i], 1], gt_vals[self.E[i], 1]])
            z = np.array([gt_vals[self.S[i], 2], gt_vals[self.E[i], 2]])
            self.plots[i][0].set_xdata(x - 1)
            self.plots[i][0].set_ydata(y)
            self.plots[i][0].set_3d_properties(z)
            self.plots[i][0].set_color(gtcolor)

        pred_vals = np.reshape(pro_channels, (24, -1))
        for i in np.arange(len(self.S)):
            # 设置关节位置
            x = np.array([pred_vals[self.S[i], 0], pred_vals[self.E[i], 0]])
            y = np.array([pred_vals[self.S[i], 1], pred_vals[self.E[i], 1]])
            z = np.array([pred_vals[self.S[i], 2], pred_vals[self.E[i], 2]])

            self.plots_IPP[i][0].set_xdata(x)
            self.plots_IPP[i][0].set_ydata(y)
            self.plots_IPP[i][0].set_3d_properties(z)
            self.plots_IPP[i][0].set_color(pcolor if pre == 1 else ocolor)
            # self.plots_IPP[i][0].set_alpha(0.7)
        gcn_vals = np.reshape(gcn_channels, (24, -1))
        for i in np.arange(len(self.S)):
            # 设置关节位置
            x = np.array([gcn_vals[self.S[i], 0], gcn_vals[self.E[i], 0]])
            y = np.array([gcn_vals[self.S[i], 1], gcn_vals[self.E[i], 1]])
            z = np.array([gcn_vals[self.S[i], 2], gcn_vals[self.E[i], 2]])

            self.plots_gcndct[i][0].set_xdata(x)
            self.plots_gcndct[i][0].set_ydata(y)
            self.plots_gcndct[i][0].set_3d_properties(z)
            self.plots_gcndct[i][0].set_color(pcolor if pre == 1 else ocolor)
            # self.plots_pred[i][0].set_alpha(0.7)
        # 设置三个轴的范围
        r = 1
        xroot, yroot, zroot = gt_vals[0, 0], gt_vals[0, 1], gt_vals[0, 2]
        self.ax.set_xlim3d([-3 + xroot, 3 + xroot])
        self.ax.set_zlim3d([-r + zroot, r + zroot])
        self.ax.set_ylim3d([-r + yroot, r + yroot])

    def update_7(self, gt_c, gcndct_c, residual_c, jinrnn_c, rnngcn_c, rnnpro_c, IPP_c, pre):
        gt_vals = np.reshape(gt_c, (24, -1))
        gtcolor = "#5a5c5e"  #灰色
        pcolor = "#AE433D"#暗红
        ocolor = "#354E87"#暗蓝
        for i in np.arange(len(self.S)):
            # 将数据转化为numpy 格式
            x = np.array([gt_vals[self.S[i], 0], gt_vals[self.E[i], 0]])
            y = np.array([gt_vals[self.S[i], 1], gt_vals[self.E[i], 1]])
            z = np.array([gt_vals[self.S[i], 2], gt_vals[self.E[i], 2]])
            self.plots[i][0].set_xdata(x - 6)
            self.plots[i][0].set_ydata(y)
            self.plots[i][0].set_3d_properties(z)
            self.plots[i][0].set_color(pcolor if pre == 1 else ocolor)

        gcn_vals = np.reshape(gcndct_c, (24, -1))
        for i in np.arange(len(self.S)):
            # 设置关节位置
            x = np.array([gcn_vals[self.S[i], 0], gcn_vals[self.E[i], 0]])
            y = np.array([gcn_vals[self.S[i], 1], gcn_vals[self.E[i], 1]])
            z = np.array([gcn_vals[self.S[i], 2], gcn_vals[self.E[i], 2]])
            self.plots_gcndct[i][0].set_xdata(x - 4)
            self.plots_gcndct[i][0].set_ydata(y)
            self.plots_gcndct[i][0].set_3d_properties(z)
            self.plots_gcndct[i][0].set_color(pcolor if pre == 1 else ocolor)
            # self.plots_pred[i][0].set_alpha(0.7)
        res_vals = np.reshape(residual_c, (24, -1))
        for i in np.arange(len(self.S)):
            # 设置关节位置
            x = np.array([res_vals[self.S[i], 0], res_vals[self.E[i], 0]])
            y = np.array([res_vals[self.S[i], 1], res_vals[self.E[i], 1]])
            z = np.array([res_vals[self.S[i], 2], res_vals[self.E[i], 2]])

            self.plots_residual[i][0].set_xdata(x - 2)
            self.plots_residual[i][0].set_ydata(y)
            self.plots_residual[i][0].set_3d_properties(z)
            self.plots_residual[i][0].set_color(pcolor if pre == 1 else ocolor)
            # self.plots_pred[i][0].set_alpha(0.7)
        jinRNN_vals = np.reshape(jinrnn_c, (24, -1))
        for i in np.arange(len(self.S)):
            # 设置关节位置
            x = np.array([jinRNN_vals[self.S[i], 0], jinRNN_vals[self.E[i], 0]])
            y = np.array([jinRNN_vals[self.S[i], 1], jinRNN_vals[self.E[i], 1]])
            z = np.array([jinRNN_vals[self.S[i], 2], jinRNN_vals[self.E[i], 2]])

            self.plots_jinRNN[i][0].set_xdata(x)
            self.plots_jinRNN[i][0].set_ydata(y)
            self.plots_jinRNN[i][0].set_3d_properties(z)
            self.plots_jinRNN[i][0].set_color(pcolor if pre == 1 else ocolor)
        rnngcn_vals = np.reshape(rnngcn_c, (24, -1))
        for i in np.arange(len(self.S)):
            # 设置关节位置
            x = np.array([rnngcn_vals[self.S[i], 0], rnngcn_vals[self.E[i], 0]])
            y = np.array([rnngcn_vals[self.S[i], 1], rnngcn_vals[self.E[i], 1]])
            z = np.array([rnngcn_vals[self.S[i], 2], rnngcn_vals[self.E[i], 2]])

            self.plots_rnngcn[i][0].set_xdata(x + 2)
            self.plots_rnngcn[i][0].set_ydata(y)
            self.plots_rnngcn[i][0].set_3d_properties(z)
            self.plots_rnngcn[i][0].set_color(gtcolor)
        rnnpro_vals = np.reshape(rnnpro_c, (24, -1))
        for i in np.arange(len(self.S)):
            # 设置关节位置
            x = np.array([rnnpro_vals[self.S[i], 0], rnnpro_vals[self.E[i], 0]])
            y = np.array([rnnpro_vals[self.S[i], 1], rnnpro_vals[self.E[i], 1]])
            z = np.array([rnnpro_vals[self.S[i], 2], rnnpro_vals[self.E[i], 2]])

            self.plots_rnnpro[i][0].set_xdata(x + 4)
            self.plots_rnnpro[i][0].set_ydata(y)
            self.plots_rnnpro[i][0].set_3d_properties(z)
            self.plots_rnnpro[i][0].set_color(pcolor if pre == 1 else ocolor)
        IPP_vals = np.reshape(IPP_c, (24, -1))
        for i in np.arange(len(self.S)):
            # 设置关节位置
            x = np.array([IPP_vals[self.S[i], 0], IPP_vals[self.E[i], 0]])
            y = np.array([IPP_vals[self.S[i], 1], IPP_vals[self.E[i], 1]])
            z = np.array([IPP_vals[self.S[i], 2], IPP_vals[self.E[i], 2]])

            self.plots_IPP[i][0].set_xdata(x + 6)
            self.plots_IPP[i][0].set_ydata(y)
            self.plots_IPP[i][0].set_3d_properties(z)
            self.plots_IPP[i][0].set_color(pcolor if pre == 1 else ocolor)
        # 设置三个轴的范围
        xroot, yroot, zroot = jinRNN_vals[0, 0], jinRNN_vals[0, 1], jinRNN_vals[0, 2]
        self.ax.set_xlim3d([-4 + xroot, 4 + xroot])
        self.ax.set_zlim3d([-0.6 + zroot, 0.6 + zroot])
        self.ax.set_ylim3d([-0.6 + yroot, 0.6 + yroot])

    def update_5(self, gt_c, gcndct_c, residual_c, rnnpro_c, IPP_c, pre):
        gt_vals = np.reshape(gt_c, (24, -1))
        gtcolor = "#5a5c5e"
        pcolor = "#AE433D"
        ocolor = "#354E87"
        for i in np.arange(len(self.S)):
            # 将数据转化为numpy 格式
            x = np.array([gt_vals[self.S[i], 0], gt_vals[self.E[i], 0]])
            y = np.array([gt_vals[self.S[i], 1], gt_vals[self.E[i], 1]])
            z = np.array([gt_vals[self.S[i], 2], gt_vals[self.E[i], 2]])
            self.plots[i][0].set_xdata(x - 4)
            self.plots[i][0].set_ydata(y)
            self.plots[i][0].set_3d_properties(z)
            self.plots[i][0].set_color(pcolor if pre == 1 else gtcolor)

        gcn_vals = np.reshape(gcndct_c, (24, -1))
        for i in np.arange(len(self.S)):
            # 设置关节位置
            x = np.array([gcn_vals[self.S[i], 0], gcn_vals[self.E[i], 0]])
            y = np.array([gcn_vals[self.S[i], 1], gcn_vals[self.E[i], 1]])
            z = np.array([gcn_vals[self.S[i], 2], gcn_vals[self.E[i], 2]])
            self.plots_gcndct[i][0].set_xdata(x - 2)
            self.plots_gcndct[i][0].set_ydata(y)
            self.plots_gcndct[i][0].set_3d_properties(z)
            self.plots_gcndct[i][0].set_color(pcolor if pre == 1 else gtcolor)
            # self.plots_pred[i][0].set_alpha(0.7)
        res_vals = np.reshape(residual_c, (24, -1))
        for i in np.arange(len(self.S)):
            # 设置关节位置
            x = np.array([res_vals[self.S[i], 0], res_vals[self.E[i], 0]])
            y = np.array([res_vals[self.S[i], 1], res_vals[self.E[i], 1]])
            z = np.array([res_vals[self.S[i], 2], res_vals[self.E[i], 2]])

            self.plots_residual[i][0].set_xdata(x)
            self.plots_residual[i][0].set_ydata(y)
            self.plots_residual[i][0].set_3d_properties(z)
            self.plots_residual[i][0].set_color(pcolor if pre == 1 else gtcolor)
            # self.plots_pred[i][0].set_alpha(0.7)
        rnnpro_vals = np.reshape(rnnpro_c, (24, -1))
        for i in np.arange(len(self.S)):
            # 设置关节位置
            x = np.array([rnnpro_vals[self.S[i], 0], rnnpro_vals[self.E[i], 0]])
            y = np.array([rnnpro_vals[self.S[i], 1], rnnpro_vals[self.E[i], 1]])
            z = np.array([rnnpro_vals[self.S[i], 2], rnnpro_vals[self.E[i], 2]])

            self.plots_rnnpro[i][0].set_xdata(x + 2)
            self.plots_rnnpro[i][0].set_ydata(y)
            self.plots_rnnpro[i][0].set_3d_properties(z)
            self.plots_rnnpro[i][0].set_color(pcolor if pre == 1 else gtcolor)
        IPP_vals = np.reshape(IPP_c, (24, -1))
        for i in np.arange(len(self.S)):
            # 设置关节位置
            x = np.array([IPP_vals[self.S[i], 0], IPP_vals[self.E[i], 0]])
            y = np.array([IPP_vals[self.S[i], 1], IPP_vals[self.E[i], 1]])
            z = np.array([IPP_vals[self.S[i], 2], IPP_vals[self.E[i], 2]])

            self.plots_IPP[i][0].set_xdata(x + 4)
            self.plots_IPP[i][0].set_ydata(y)
            self.plots_IPP[i][0].set_3d_properties(z)
            self.plots_IPP[i][0].set_color(pcolor if pre == 1 else gtcolor)
        # 设置三个轴的范围
        xroot, yroot, zroot = res_vals[0, 0], res_vals[0, 1], res_vals[0, 2]
        self.ax.set_xlim3d([-4 + xroot, 4 + xroot])
        self.ax.set_zlim3d([-0.6 + zroot, 0.6 + zroot])
        self.ax.set_ylim3d([-0.6 + yroot, 0.6 + yroot])


def plot_predictions(xyz_gt, xyz_pro, fig, ax, f_title, pig_name, ):
    nframes_pred = xyz_pro.shape[0]
    # === Plot and animate ===
    ob = Ax3DPose(ax)
    # Plot the prediction
    for i in range(nframes_pred):
        ob.update(xyz_gt[i, :], xyz_pro[i, :])

        ax.set_title(f_title + ' frame:{:d}'.format(i + 1), loc="left")
        plt.savefig(pig_name + ' frame{:d}'.format(i + 1) + ".png")
        plt.show(block=False)
        fig.canvas.draw()
        plt.pause(0.1)


def plot_predictions_3(xyz_gt, xyz_pro, xyz_gcn, fig, ax, f_title, pig_name):
    nframes_pred = xyz_pro.shape[0]
    # === Plot and animate ===
    ob = Ax3DPose(ax)
    # Plot the prediction
    pre = 0
    for i in range(nframes_pred):
        # if i > 17:
        #     pre = 1
        # else:
        #     pre = 0
        ob.update_3(xyz_gt[i, :], xyz_pro[i, :], xyz_gcn[i, :], pre)

        ax.set_title(f_title + ' frame:{:d}'.format(i + 1), loc="left")
        plt.savefig(pig_name + ' frame{:d}'.format(i + 1) + ".png")
        plt.show(block=False)
        fig.canvas.draw()
        plt.pause(0.001)


def plot_predictions_n(xyz_1, xyz_2, xyz_3, xyz_4, xyz_5, fig, ax, f_title,
                       pig_name, pre):
    nframes_pred = xyz_1.shape[0]
    # === Plot and animate ===
    ob = Ax3DPose(ax)
    # Plot the prediction
    for i in range(nframes_pred):
        ob.update_5(xyz_1[i], xyz_2[i], xyz_3[i], xyz_4[i], xyz_5[i], pre)
        ax.set_title(f_title + ' frame:{:d}'.format(i + 1), loc="left")
        plt.savefig(pig_name + ' frame{:d}'.format(i + 1) + ".png")
        plt.show(block=False)
        fig.canvas.draw()
        plt.pause(0.001)


def plot_gt(xyz_gt, fig, ax, f_title, pig_name, input_color):
    nframes_gt = xyz_gt.shape[0]
    # === Plot and animate ===
    ob = Ax3DPose(ax)
    # Plot the prediction
    for i in range(nframes_gt):
        ob.up_x(xyz_gt[i, :], input_color)
        ax.set_title(f_title + ' frame:{:d}'.format(i + 1), loc="left")
        plt.savefig(pig_name + ' frame{:d}'.format(i + 1) + ".png")
        plt.show(block=False)
        fig.canvas.draw()
        plt.pause(0.001)

# def plot_vertices(poses_gt, poses_pred, fig, ax, f_tittle):
#     imw, imh = 1600, 1600
#     mv = MeshViewer(width=imw, height=imh, use_offscreen=True)
#     bm = BodyModel(bm_fname=bm_fname, num_betas=num_betas).to(comp_device)
#     faces = c2c(bm.f)
