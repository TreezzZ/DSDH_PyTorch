import torch
import torchvision.transforms as transforms

import os

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from PIL import Image
from data.transform import encode_onehot, Onehot


def load_data(root, batch_size, workers):
    """
    Load imagenet dataset

    Args:
        root (str): Path of imagenet dataset.
        batch_size (int): Number of samples in one batch.
        workers (int): Number of data loading threads.

    Returns:
        train_dataloader (torch.utils.data.dataloader.DataLoader): Training dataset loader.
        query_dataloader (torch.utils.data.dataloader.DataLoader): Query dataset loader.
        retrieval_dataloader (torch.utils.data.dataloader.DataLoader): Validation dataset loader.
    """
    # Data transform
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

    # Construct data loader
    train_dir = os.path.join(root, 'train')
    query_dir = os.path.join(root, 'query')
    retrieval_dir = os.path.join(root, 'database')

    train_dataset = ImagenetDataset(
        train_dir,
        transform=train_transform,
        target_transform=Onehot(100),
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
    )

    query_dataset = ImagenetDataset(
        query_dir,
        transform=test_transform,
        target_transform=Onehot(100),
    )

    query_dataloader = DataLoader(
        query_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )

    retrieval_dataset = ImagenetDataset(
        retrieval_dir,
        transform=test_transform,
        target_transform=Onehot(100),
    )

    retrieval_dataloader = DataLoader(
        retrieval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )

    return train_dataloader, query_dataloader, retrieval_dataloader


class ImagenetDataset(Dataset):
    classes = None
    class_to_idx = None

    def __init__(self, root, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.imgs = []
        self.targets = []

        # Assume file alphabet order is the class order
        if ImagenetDataset.class_to_idx is None:
            ImagenetDataset.classes, ImagenetDataset.class_to_idx = self._find_classes(root)

        for i, cl in enumerate(ImagenetDataset.classes):
            cur_class = os.path.join(self.root, cl)
            files = os.listdir(cur_class)
            files = [os.path.join(cur_class, i) for i in files]
            self.imgs.extend(files)
            self.targets.extend([ImagenetDataset.class_to_idx[cl] for i in range(len(files))])
        self.targets = torch.tensor(self.targets)
        self.onehot_targets = torch.from_numpy(encode_onehot(self.targets, 100)).float()

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, item):
        img, target = self.imgs[item], self.targets[item]

        img = Image.open(img).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, item

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def get_onehot_targets(self):
        """
        Return one-hot encoding targets.
        """
        return self.onehot_targets

