# -*- coding:utf-8 -*-

from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from data.transform import img_transform

import numpy as np
import os
from PIL import Image


def load_data(opt):
    """加载NUS-WIDE数据集

    Parameters
        opt:Parser
        配置

    Returns
        query_dataloader, train_dataloader, database_dataloader: DataLoader
        数据加载器
    """
    NUS_WIDE.init(opt.data_path, opt.num_query, opt.num_train)
    query_dataset = NUS_WIDE('query', transform=img_transform())
    train_dataset = NUS_WIDE('train', transform=img_transform())
    database_dataset = NUS_WIDE('all', transform=img_transform())

    query_dataloader = DataLoader(query_dataset,
                                  batch_size=opt.batch_size,
                                  num_workers=opt.num_workers,
                                  )
    train_dataloader = DataLoader(train_dataset,
                                  shuffle=True,
                                  batch_size=opt.batch_size,
                                  num_workers=opt.num_workers,
                                  )
    database_dataloader = DataLoader(database_dataset,
                                     batch_size=opt.batch_size,
                                     num_workers=opt.num_workers,
                                     )

    return query_dataloader, train_dataloader, database_dataloader


class NUS_WIDE(Dataset):
    @staticmethod
    def init(path, num_query, num_train):
        # load data, tags
        NUS_WIDE.ALL_IMG = np.load(os.path.join(path, 'nus-wide-21-img.npy'))
        NUS_WIDE.ALL_TARGETS = np.load(os.path.join(path, 'nus-wide-21-tag.npy')).astype(np.float32)
        NUS_WIDE.ALL_IMG = NUS_WIDE.ALL_IMG.transpose((0, 2, 3, 1))

        # 打算平衡采样，每类都采样一些，可是内存大小不够
        # split data, tags
        # query_per_class = num_query // 21
        # train_per_class = num_train // 21
        # for i in range(21):
        #     non_zero_index = np.asarray(np.where(NUS_WIDE.ALL_TAGS[:, i] == 1))
        #     non_zero_index = non_zero_index[np.random.permutation(non_zero_index.shape[0])]
        #     if not i:
        #         query_index = non_zero_index[:query_per_class]
        #         train_index = non_zero_index[query_per_class: query_per_class + train_per_class]
        #     else:
        #         query_index = np.hstack((query_index, non_zero_index[:query_per_class]))
        #         train_index = np.hstack(
        #             (train_index, non_zero_index[query_per_class: query_per_class + train_per_class]))

        # split data, tags
        perm_index = np.random.permutation(NUS_WIDE.ALL_IMG.shape[0])
        query_index = perm_index[:num_query]
        train_index = perm_index[:num_train]

        NUS_WIDE.QUERY_IMG = NUS_WIDE.ALL_IMG[query_index, :]
        NUS_WIDE.QUERY_TARGETS = NUS_WIDE.ALL_TARGETS[query_index, :]
        NUS_WIDE.TRAIN_IMG = NUS_WIDE.ALL_IMG[train_index, :]
        NUS_WIDE.TRAIN_TARGETS = NUS_WIDE.ALL_TARGETS[train_index, :]

    def __init__(self, mode, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

        if mode == 'train':
            self.img = NUS_WIDE.TRAIN_IMG
            self.targets = NUS_WIDE.TRAIN_TARGETS
        elif mode == 'query':
            self.img = NUS_WIDE.QUERY_IMG
            self.targets = NUS_WIDE.QUERY_TARGETS
        else:
            self.img = NUS_WIDE.ALL_IMG
            self.targets = NUS_WIDE.ALL_TARGETS

    def __getitem__(self, index):
        img, target = self.img[index], self.targets[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return self.img.shape[0]
