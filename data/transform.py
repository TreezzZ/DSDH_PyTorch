import numpy as np
import torch
import torchvision.transforms as transforms


def encode_onehot(labels, num_classes=10):
    """
    one-hot labels

    Args:
        labels (numpy.ndarray): labels.
        num_classes (int): Number of classes.

    Returns:
        onehot_labels (numpy.ndarray): one-hot labels.
    """
    onehot_labels = np.zeros((len(labels), num_classes))

    for i in range(len(labels)):
        onehot_labels[i, labels[i]] = 1

    return onehot_labels


class Onehot(object):
    def __init__(self, num_classes=10):
        self.num_classes = num_classes

    def __call__(self, sample):
        target_onehot = torch.zeros(self.num_classes)
        target_onehot[sample] = 1

        return target_onehot


def train_transform():
    """
    Training images transform.

    Args
        None

    Returns
        transform(torchvision.transforms): transform
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.RandomResizedCrop(224),                         
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])


def query_transform():
    """
    Query images transform.

    Args
        None

    Returns
        transform(torchvision.transforms): transform
    """
    # Data transform
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
