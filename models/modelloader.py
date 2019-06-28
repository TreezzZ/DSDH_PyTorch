# -*- coding:utf-8 -*-

import models.alexnet as alexnet


def load_model(name, pretrained=True, num_classes=None):
    """加载模型

    Parameters
        name: str
        模型名称

        pretrained: bool
        True: 加载预训练模型; False: 加载未训练模型

        num_classes: int
        CNN最后一层输出类别

    Returns
        model: model
        模型
    """
    if name == 'alexnet':
        return alexnet.load_model(pretrained=pretrained, num_classes=num_classes)
