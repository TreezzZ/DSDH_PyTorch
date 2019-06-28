#!/usr/bin/env python
# -*- coding: utf-8 -*-

import data.cifar10 as cifar10
import data.nus_wide as nus_wide


def load_data(opt):
    """加载数据

    Parameters
        opt: Parser
        参数

    Returns
        DataLoader
        数据加载器
    """
    if opt.dataset == 'cifar10':
        return cifar10.load_data(opt)
    elif opt.dataset == 'nus-wide':
        return nus_wide.load_data(opt)
