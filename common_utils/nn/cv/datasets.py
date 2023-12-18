from tqdm import tqdm
import glob
import pandas as pd
import random
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Dataset
import cv2
import torch
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform


def mixup(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
    """
    https://www.kaggle.com/code/phalanx/train-swin-t-pytorch-lightning
    """
    assert alpha > 0, "alpha should be larger than 0"
    assert x.size(0) > 1, "Mixup cannot be applied to a single instance."

    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x.size()[0])
    mixed_x = lam * x + (1 - lam) * x[rand_index, :]
    target_a, target_b = y, y[rand_index]
    return mixed_x, target_a, target_b, lam


"""
def get_train_aug(cfg):
    p = 0.5

    return [
        # A.Resize(cfg.size, cfg.size),
        A.RandomResizedCrop(cfg.size, cfg.size, scale=(0.85, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0, scale_limit=0,
                           rotate_limit=15, p=p),
        # A.OneOf([
        #     A.GaussNoise(var_limit=[10, 50]),
        #     A.GaussianBlur(),
        #     A.MotionBlur(),
        #     A.MedianBlur(),
        # ], p=0.4),
        # A.Cutout(p=p),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ]

def get_val_aug(cfg):

    return [
        A.Resize(cfg.size, cfg.size),
        A.Normalize(
            mean=cfg.aug_mean,
            std=cfg.aug_std,
        ),
        ToTensorV2(),
    ]
"""

"""
def get_transforms(data, cfg):
    if data == 'train':
        return A.Compose(get_train_aug(cfg))

    elif data == 'valid':
        return A.Compose(get_val_aug(cfg))
"""

"""
def get_transforms(data, cfg):
    if data == 'train':
        print('get_train_aug')
        print(cfg.train_aug_path)
        if hasattr(cfg, 'train_aug_path'):
            aug = A.load(cfg.comp_dir_path +
                         cfg.train_aug_path, data_format='yaml')
        else:
            aug = A.Compose(get_train_aug(cfg))

    elif data == 'valid':
        print('get_val_aug')

        if hasattr(cfg, 'valid_aug_path'):
            aug = A.load(cfg.comp_dir_path +
                         cfg.valid_aug_path, data_format='yaml')
        else:
            aug = A.Compose(get_val_aug(cfg))

    # print(aug)
    return aug
"""
def get_transforms(data, cfg):
    if data == 'train':
        aug = A.Compose(cfg.train_aug_list)
    elif data == 'valid':
        aug = A.Compose(cfg.valid_aug_list)

    # print(aug)
    return aug

class CustomDataset(Dataset):
    def __init__(self, df, cfg, labels=None, transform=None):
        self.df = df
        self.cfg = cfg
        self.file_paths = df['file_path'].values
        # self.labels = df[self.cfg.target_col].values
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def read_image(self, idx):
        file_path = self.file_paths[idx]

        if self.cfg.image_file_suffix == '.npy':
            # image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            image = np.load(file_path)
        else:
            if self.cfg.in_chans == 1:
                image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            else:
                image = cv2.imread(file_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    def __getitem__(self, idx):
        image = self.read_image(idx)

        if self.transform:
            image = self.transform(image=image)['image']

        if self.labels is None:
            return image

        if self.cfg.objective_cv == 'multiclass':
            label = torch.tensor(self.labels[idx]).long()
        else:
            label = torch.tensor(self.labels[idx]).float()

        return image, label


'''
def plot_aug(train, cfg, plot_count=10):
    file_paths = train['file_path'].values
    file_paths = file_paths[:plot_count]
    # transform = get_transforms(data='train', cfg=cfg)

    """
    aug_list = get_train_aug(cfg)
    transform = A.Compose(
        [t for t in aug_list if not isinstance(t, (A.Normalize, ToTensorV2))])
    """

    transform = get_transforms('train', cfg)
    transform = A.Compose(
        [t for t in transform if not isinstance(t, (A.Normalize, ToTensorV2))])

    for i, file_path in enumerate(file_paths):

        if i == plot_count:
            break

        # image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.imread(file_path)
        print(image.shape)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        aug_image = transform(image=image)['image']

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        axes[0].imshow(image, cmap="gray")
        axes[1].imshow(aug_image, cmap="gray")
        plt.savefig(cfg.figures_dir + f'aug_{i}.png')
'''

def plot_aug(train, cfg, plot_count=10):
    dataset = CustomDataset(train, cfg)

    transform = cfg.train_aug_list
    transform = A.Compose(
        [t for t in transform if not isinstance(t, (A.Normalize, ToTensorV2))])

    for i in range(plot_count):

        image = dataset[i]
        aug_image = transform(image=image)['image']

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        axes[0].imshow(image, cmap="gray")
        axes[1].imshow(aug_image, cmap="gray")
        plt.savefig(cfg.figures_dir + f'aug_{i}.png')
