# -*- coding:utf-8 -*-

import torchvision
import torch.nn as nn


def load_model(pretrained=True, num_classes=None):
    """加载model

    Parameters
        pretrained: bool
        True: 加载预训练模型; False: 加载未训练模型

        num_classes: int
        Alexnet最后一层输出

    Returns
        alexnet_model: model
        CNN模型
    """
    model = torchvision.models.alexnet(pretrained=pretrained)

    if pretrained:
        fc1 = nn.Linear(256 * 6 * 6, 4096)
        fc1.weight = model.classifier[1].weight
        fc1.bias = model.classifier[1].bias

        fc2 = nn.Linear(4096, 4096)
        fc2.weight = model.classifier[4].weight
        fc2.bias = model.classifier[4].bias

        classifier = nn.Sequential(
            nn.Dropout(),
            fc1,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            fc2,
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        model.classifier = classifier

    return model
