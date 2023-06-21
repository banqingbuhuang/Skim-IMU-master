#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import os


class AccumLoss(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


def lr_decay(optimizer, lr_now, gamma):  # learning rate decay
    lr = lr_now * gamma
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


# 文件名会传进来的script_name
def save_ckpt(state, ckpt_path,  file_name=['ckpt_best.pth.tar']):
    file_path = os.path.join(ckpt_path, file_name[0])
    torch.save(state, file_path)
