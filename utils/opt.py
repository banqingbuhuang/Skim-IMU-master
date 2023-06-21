import os
import argparse
from pprint import pprint

import torch

"""
DIP_IMU
0:head 
1:spine2, 
2:belly,
3:lchest, 
4:rchest, 
5:lshoulder, 
6:rshoulder, 
7:lelbow, 
8:relbow, 
9:lhip, 
10:rhip, 
11:lknee, 
12:rknee,
13:lwrist,
14:rwrist, 
15:lankle, 
16:rankle
"""
"""
Transpose选择：7, 8, 11, 12, 0, 2
lelbow,relbow,lknee,rknee,head,spine2

"""


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = None

    def _initial(self):
        # ===============================================================
        #                     General options
        # ===============================================================

        self.parser.add_argument('--smpl_file', type=str,
                                 default='/data/xt/body_models/vPOSE/models/smpl/SMPL_MALE.pkl',
                                 help='path to SMPL model')

        self.parser.add_argument('--device', type=str, default='cuda:0',
                                 help='path to amass Synthetic dataset')
        self.parser.add_argument('--data_dip_total_short_dir', type=str, default='/data/wwu/xt/dataset/dip_total_short',
                                 help='path to DIP_IMU dataset')
        self.parser.add_argument('--data_dip_total_dir', type=str, default='/data/wwu/xt/dataset/dip_total',
                                 help='path to DIP_IMU dataset')
        self.parser.add_argument('--data_xt_dip_total_dir', type=str, default='/data/xt/dataset/dip_total',
                                 help='path to DIP_IMU dataset')
        self.parser.add_argument('--data_xt_amassdip_dir', type=str, default='/data/xt/dataset/AMASS_DIP_2',
                                 help='path to DIP_IMU dataset')
        self.parser.add_argument('--data_xt_dip_dir', type=str, default='/data/xt/dataset/only_dip',
                                 help='path to DIP_IMU dataset')
        self.parser.add_argument('--data_dip_dir', type=str, default='/data/wwu/xt/dataset/only_dip',
                                 help='path to DIP_IMU dataset')
        self.parser.add_argument('--data_benji_dip_dir', type=str, default='C:/Gtrans/dataset/dataset/only_dip',
                                 help='path to DIP_IMU dataset')
        self.parser.add_argument('--data_total_dir', type=str, default='/data/wwu/xt/dataset/only_total',
                                 help='path to DIP_IMU dataset')
        self.parser.add_argument('--data_xt_total_dir', type=str, default='/data/xt/dataset/only_total',
                                 help='path to DIP_IMU dataset')
        # self.parser.add_argument('--data_dir_cmu', type=str, default='D://cmu_mocap/', help='path to CMU dataset')
        self.parser.add_argument('--exp', type=str, default='test', help='ID of experiment')
        self.parser.add_argument('--ckpt', type=str, default='checkpoint/', help='path to save checkpoint')

        # ===============================================================
        #                     Model options
        # ===============================================================
        self.parser.add_argument('--max_norm', dest='max_norm', action='store_true',
                                 help='maxnorm constraint to weights')
        self.parser.add_argument('--linear_size', type=int, default=256, help='size of each model layer')
        self.parser.add_argument('--num_stage', type=int, default=12, help='# layers in linear model')

        # ===============================================================
        #                     Running options
        # ===============================================================
        self.parser.add_argument('--lr', type=float, default=1.0e-4)
        self.parser.add_argument('--lr_decay', type=int, default=2, help='every lr_decay epoch do lr decay')
        self.parser.add_argument('--lr_gamma', type=float, default=0.96)
        self.parser.add_argument('--input_n', type=int, default=36, help='observed seq length')
        self.parser.add_argument('--output_n', type=int, default=24, help='future seq length')
        self.parser.add_argument('--all_n', type=int, default=60, help='number of DCT coeff. preserved for 3D')
        self.parser.add_argument('--actions', type=str, default='all', help='path to save checkpoint')
        self.parser.add_argument('--epochs', type=int, default=50)
        self.parser.add_argument('--dropout', type=float, default=0.5,
                                 help='dropout probability, 1.0 to make no dropout')
        self.parser.add_argument('--train_batch', type=int, default=128)
        self.parser.add_argument('--test_batch', type=int, default=128)
        self.parser.add_argument('--job', type=int, default=4, help='subprocesses to use for data loading')
        self.parser.add_argument('--is_load', dest='is_load', action='store_true', help='wether to load existing model')
        self.parser.add_argument('--sample_rate', type=int, default=2, help='frame sampling rate')
        self.parser.add_argument('--is_norm_dct', dest='is_norm_dct', action='store_true',
                                 help='whether to normalize the dct coeff')
        self.parser.add_argument('--is_norm', dest='is_norm', action='store_true',
                                 help='whether to normalize the angles/3d coordinates')

        """
        8个位置包括：0：根节点，1：头，2：左手腕，3：右手腕，4：左膝盖，5：右膝盖，6：左脚踝，7：右脚踝
        """
        adje_8 = [[0, 1, 0, 0, 1, 1, 0, 0],
                  [1, 0, 1, 1, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0, 0, 0],
                  [1, 0, 0, 0, 0, 0, 1, 0],
                  [1, 0, 0, 0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0]
                  ]
        adje_9_part = [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 1],
                       [0, 0, 0, 0, 0, 0, 0, 0, 1],
                       [0, 0, 0, 0, 0, 0, 0, 0, 1],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 1, 1, 0, 0, 0, 0, 0]]
        adje_9_all = [[0, 1, 0, 0, 1, 1, 0, 0, 0],
                      [1, 0, 0, 1, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0, 0, 0, 1],
                      [0, 1, 0, 0, 0, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0, 0, 1, 0, 0],
                      [1, 0, 0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0, 0, 0],
                      [0, 1, 1, 0, 0, 0, 0, 0, 0]]
        adje_10_part = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                        [0, 1, 1, 1, 0, 0, 0, 0, 1, 0]
                        ]
        adje_10_all = [[0, 1, 0, 0, 1, 1, 0, 0, 0, 0],
                       [1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                       [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                       [1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                       [1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                       [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                       [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 1, 0, 0, 0, 0, 0, 0]
                       ]

        """
        六个位置的关节包括：0:左手，1:右手，2：左膝盖，3：右膝盖，4：头，5：根节点
        """
        adj_6 = [[0, 0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [1, 1, 0, 0, 0, 1],
                 [0, 0, 1, 1, 1, 0]]
        adj_7_1 = [[0, 0, 0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0, 0, 0],
                   [1, 1, 0, 0, 1, 0, 0]]
        adj_7_2 = [[0, 0, 0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0, 1, 0],
                   [0, 1, 0, 0, 0, 1, 1],
                   [0, 0, 1, 1, 1, 0, 0],
                   [1, 0, 0, 0, 1, 0, 0]]
        adj_8_1 = [[0, 0, 0, 0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 1],
                   [1, 1, 0, 0, 1, 0, 1, 0]]
        adj_8_2 = [[0, 0, 0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 0, 1, 1, 1],
                   [0, 0, 1, 1, 1, 0, 0, 0],
                   [1, 0, 0, 0, 1, 0, 0, 0],
                   [0, 1, 0, 0, 1, 0, 0, 0]]
        adj_9_1 = [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 1, 1, 0, 0, 0, 0, 0],
                   ]
        adj_9_2 = [[0, 0, 0, 0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0, 1, 0, 0, 1],
                   [0, 0, 0, 0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 0, 0, 1, 1, 1, 0],
                   [0, 0, 1, 1, 1, 0, 0, 0, 0],
                   [1, 0, 0, 0, 1, 0, 0, 0, 0],
                   [0, 1, 0, 0, 1, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0, 0, 0, 0, 0],
                   ]
        adj_10_1 = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 1, 1, 0, 0, 0, 0, 1, 0]]
        adj_10_2 = [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
                    [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]]
        # 最后的二十四个关节的顺序是按照smpl模型的顺序来定义的，
        #          0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23
        adj_24 = [[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0
                  [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1
                  [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 2
                  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 3
                  [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 4
                  [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 5
                  [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 6
                  [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 7
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 8
                  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 9
                  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 10
                  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 11
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # 12
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 13
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 14
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 15
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 16
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 17
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],  # 18
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],  # 19
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],  # 20
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],  # 21
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # 22
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  # 23
                  ]
        adjs = [adj_6, adj_7_1, adj_7_2, adj_8_1, adj_8_2, adj_9_1, adj_9_2, adj_10_1, adj_10_2, adj_24]
        adjs_8 = [adje_8, adje_9_part, adje_9_all, adje_10_part, adje_10_all, adj_24]

        loss_weight = [2, 1, 1, 1, 3, 3, 1, 4, 4, 1, 3, 3, 1, 1, 1, 2, 1, 1, 3, 3, 4, 4, 3, 3]  # 0
        loss_weight = torch.as_tensor(loss_weight)
        ignore = [0, 7, 8, 10, 11, 20, 21, 22, 23]
        ignore = torch.as_tensor(ignore)
        self.parser.add_argument('--loss_weight', type=list, default=loss_weight, help='all the Adjacency Matrix s')
        self.parser.add_argument('--ignore', type=list, default=ignore, help='the list of ignore joints')
        self.parser.add_argument('--adjs', type=list, default=adjs, help='all the Adjacency Matrix s')
        self.parser.add_argument('--adjs_8', type=list, default=adjs_8, help='all the Adjacency Matrix s')
        self.parser.set_defaults(max_norm=True)
        self.parser.set_defaults(is_load=False)
        # self.parser.set_defaults(is_norm_dct=True)
        # self.parser.set_defaults(is_norm=True)

    def _print(self):
        print("\n==================Options=================")
        pprint(vars(self.opt), indent=4)
        print("==========================================\n")

    def parse(self):
        self._initial()
        self.opt = self.parser.parse_args()
        # do some pre-check
        ckpt = os.path.join(self.opt.ckpt, self.opt.exp)
        if not os.path.isdir(ckpt):
            os.makedirs(ckpt)
        self.opt.ckpt = ckpt
        return self.opt
