import numpy
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader

import numpy as np
import pandas
import os

from utils.amass_6_node import amass_rnn as amass
import separate.model.rnn_6_1part as nnmodel
from utils.opt import Options
from utils import loss_func, utils_utils as utils_utils

# 根节点,头，左手，右手，左膝盖，右膝盖，
IMU_mask_1 = [0, 15, 21, 20, 5, 4]


def main(opt):
    input_n = opt.input_n
    output_n = opt.output_n
    all_n = opt.all_n
    loss_weight = opt.loss_weight
    adjs = opt.adjs
    start_epoch = 00
    err_best = 10000
    lr_now = opt.lr
    batch_size = opt.train_batch
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    script_name = os.path.basename(__file__).split('.')[0]
    # 0:产生了很多的nan
    # 1:改变激励函数，改tanh为relu
    # 2:激励函数改成tanh  用tanh loss下降的速度明显快了很多
    script_name += "1_in{:d}_out{:d}_dctn{:d}".format(opt.input_n, opt.output_n, opt.all_n)
    print(">>> creating model")
    model = nnmodel.RNN_pos(input_size=72, output_size=18, hidden_size=256,
                            batch_size=batch_size, num_layers=3, device=device, dropout=0.5)
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        model = model.to(device)
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    # data loading
    print(">>>  err_best", err_best)
    print(">>> loading data")
    # data_amass_dir
    # data_benji_dir
    # data_amass_xt_dir
    train_dataset = amass(path_to_data=opt.data_amass_xt_dir, input_n=input_n, output_n=output_n,
                          split=0)
    val_dataset = amass(path_to_data=opt.data_amass_xt_dir, input_n=input_n, output_n=output_n,
                        split=1)
    test_dataset = amass(path_to_data=opt.data_amass_xt_dir, input_n=input_n, output_n=output_n,
                         split=2)
    # test_dataset = amass(path_to_data=opt.data_amass_dir, input_n=input_n, output_n=output_n,
    #                    sample_rate=sample_rate, split=2)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,  # batch 32
        shuffle=True,  # 在每个epoch开始的时候，对数据进行重新排序
        num_workers=opt.job,  # 原来写的是opt.job=10，现在因为虚拟内存不够，更改为0。
        # 这个参数决定了有几个进程来处理data loading。0意味着所有的数据都会被load进主进程。（默认为0）
        pin_memory=True)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,  # 128
        shuffle=False,
        num_workers=opt.job,  # 原来写的是opt.job=10，现在因为虚拟内存不够，更改为0。
        pin_memory=True)

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,  # 128
        shuffle=False,
        num_workers=opt.job,  # 原来写的是opt.job=10，现在因为虚拟内存不够，更改为0。
        pin_memory=True)

    print(">>> data loaded !")
    for epoch in range(start_epoch, opt.epochs):
        if (epoch + 1) % opt.lr_decay == 0:  # lr_decay=2学习率延迟
            lr_now = utils_utils.lr_decay(optimizer, lr_now, opt.lr_gamma)  # lr_gamma学习率更新倍数0.96
        print('=====================================')
        print('>>> epoch: {} | lr: {:.6f}'.format(epoch + 1, lr_now))
        ret_log = np.array([epoch + 1])
        head = np.array(['epoch'])
        # per epoch
        Ir_now, t_l, t_pos, t_bone = train(train_loader, model, optimizer, cuda=device, loss_weight=loss_weight,
                                           lr_now=lr_now, max_norm=opt.max_norm, all_n=all_n)
        print("epoch结束")
        print("epoch结束")
        print("train_loss:", t_l, "position_loss:", t_pos, "bone_loss:", t_bone)
        ret_log = np.append(ret_log, [lr_now, t_l, t_pos, t_bone])
        head = np.append(head, ['lr', 't_l', 't_pos', 't-bone'])
        # validation
        v_p, v_bone = val(val_loader, model, cuda=device, loss_weight=loss_weight, all_n=all_n)
        print("val_loss:", v_p)
        ret_log = np.append(ret_log, [v_p, v_bone])
        head = np.append(head, ['v_p', 'v_bone'])
        if not np.isnan(v_p):  # 判断空值 只有数组数值运算时可使用如果v_e不是空值
            is_best = v_p < err_best  # err_best=10000
            err_best = min(v_p, err_best)
        else:
            is_best = False
        ret_log = np.append(ret_log, is_best)  # 内容
        head = np.append(head, ['is_best'])  # 表头
        df = pandas.DataFrame(np.expand_dims(ret_log, axis=0))  # DataFrame是Python中Pandas库中的一种数据结构，它类似excel，是一种二维表。
        if epoch == start_epoch:
            df.to_csv(opt.ckpt + '/' + script_name + '.csv', header=head, index=False)
        else:
            with open(opt.ckpt + '/' + script_name + '.csv', 'a') as f:
                df.to_csv(f, header=False, index=False)
        if is_best:
            file_name = ['ckpt_' + str(script_name) + '_best.pth.tar', 'ckpt_']
            utils_utils.save_ckpt({'epoch': epoch + 1,
                                   'lr': lr_now,
                                   # 'err': test_e[0],
                                   'state_dict': model.state_dict(),
                                   'optimizer': optimizer.state_dict()},
                                  ckpt_path=opt.ckpt,
                                  file_name=file_name)


def train(train_loader, model, optimizer, cuda, loss_weight, lr_now, max_norm, all_n):
    print("进入train")
    # 初始化
    t_l = utils_utils.AccumLoss()
    t_pos = utils_utils.AccumLoss()
    t_bone = utils_utils.AccumLoss()
    # 固定句式 在训练模型时会在前面加上
    model.train()
    for i, (input_ori, input_acc, out_poses, out_joints) in enumerate(train_loader):
        if torch.isnan(input_ori).sum() == 0 and torch.isnan(input_acc).sum() == 0 \
                and torch.isnan(out_poses).sum() == 0 and torch.isnan(out_joints).sum() == 0:
            batch_size = input_ori.shape[0]  # 16
            if batch_size != 32:
                continue
            if torch.cuda.is_available():
                input_ori = input_ori.to(cuda).float()
                input_acc = input_acc.to(cuda).float()
                out_poses = out_poses.to(cuda).float()
                out_joints = out_joints[:, :, IMU_mask_1].to(cuda).float()
            # model要改
            y_out = model(input_ori, input_acc)
            loss = loss_func.position_loss(y_out, out_joints)
            optimizer.zero_grad()  # 把梯度置零，也就是把loss关于weight的导数变成0.
            loss.backward()

            if max_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()  # 则可以用所有Variable的grad成员和lr的数值自动更新Variable的数值
            n = batch_size
            t_l.update(loss.cpu().data.numpy() * n, n)
            t_pos.update(loss.cpu().data.numpy() * n, n)
            t_bone.update(loss.cpu().data.numpy() * n, n)
    return lr_now, t_l.avg, t_pos.avg, t_bone.avg


def val(train_loader, model, cuda, loss_weight, all_n):
    print("进入val")
    t_posi = utils_utils.AccumLoss()
    t_bone = utils_utils.AccumLoss()
    model.eval()
    for i, (input_ori, input_acc, out_poses, out_joints) in enumerate(train_loader):
        if torch.isnan(input_ori).sum() == 0 and torch.isnan(input_acc).sum() == 0 \
                and torch.isnan(out_poses).sum() == 0 and torch.isnan(out_joints).sum() == 0:
            batch_size = input_ori.shape[0]  # 16
            if batch_size != 32:
                continue
            if torch.cuda.is_available():
                input_ori = input_ori.to(cuda).float()
                input_acc = input_acc.to(cuda).float()
                out_poses = out_poses.to(cuda).float()
                out_joints = out_joints[:, :, IMU_mask_1].to(cuda).float()
            # model要改
            y_out = model(input_ori, input_acc)
            loss = loss_func.position_loss(y_out, out_joints)
            n = batch_size
            t_posi.update(loss.cpu().data.numpy() * n, n)
            t_bone.update(loss.cpu().data.numpy() * n, n)
    return t_posi.avg, t_bone.avg


if __name__ == "__main__":
    option = Options().parse()
    main(option)
