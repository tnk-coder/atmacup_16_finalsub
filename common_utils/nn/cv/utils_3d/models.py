import numpy as np
import torch.nn as nn
import timm
import os
import torch
import cv2
import torchvision.models as models


class BasicStem(nn.Sequential):
    """The default conv-batchnorm-relu stem
    """

    def __init__(self, in_chans):
        super(BasicStem, self).__init__(
            nn.Conv3d(in_chans, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2),
                      padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))

class R2Plus1dStem(nn.Sequential):
    """R(2+1)D stem is different than the default one as it uses separated 3D convolution
    """

    def __init__(self, in_chans):
        super(R2Plus1dStem, self).__init__(
            nn.Conv3d(in_chans, 45, kernel_size=(1, 7, 7),
                      stride=(1, 2, 2), padding=(0, 3, 3),
                      bias=False),
            nn.BatchNorm3d(45),
            nn.ReLU(inplace=True),
            nn.Conv3d(45, 64, kernel_size=(3, 1, 1),
                      stride=(1, 1, 1), padding=(1, 0, 0),
                      bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))


class CustomModel3D(nn.Module):
    def __init__(self, cfg, pretrained=False, target_size=None, model_3d_name=None):
        super().__init__()
        self.cfg = cfg

        in_chans = self.cfg.in_chans

        if model_3d_name is None:
            model_3d_name = self.cfg.model_3d_name

        self.n_features = 512
        self.target_size = self.cfg.target_size if target_size is None else target_size

        if model_3d_name == 'resnet18-3d':
            self.model = models.video.r3d_18(pretrained=pretrained)

            if in_chans != 3:
                self.model.stem = BasicStem(in_chans=in_chans)
            # self.n_features = self.model.num_features

        elif model_3d_name == 'mc3_18':
            self.model = models.video.mc3_18(pretrained=pretrained)

            if in_chans != 3:
                self.model.stem = BasicStem(in_chans=in_chans)

        elif model_3d_name == 'r2plus1d_18':
            self.model = models.video.r2plus1d_18(pretrained=pretrained)

            if in_chans != 3:
                self.model.stem = R2Plus1dStem(in_chans=in_chans)

        self.model.fc = nn.Sequential(
            nn.Linear(self.n_features, self.cfg.target_size)
        )

    def feature(self, image):

        feature = self.model(image)
        return feature

    def forward(self, image):
        """
        feature = self.feature(image)
        output = self.fc(feature)
        return output
        """
        return self.feature(image)
