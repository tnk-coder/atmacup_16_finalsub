import numpy as np
import torch.nn as nn
import timm
import os
import torch
import cv2

class CustomModel(nn.Module):
    def __init__(self, cfg, pretrained=False, target_size=None, model_name=None):
        super().__init__()
        # self.cfg = cfg

        if model_name is None:
            model_name = cfg.model_name

        print(f'pretrained: {pretrained}')

        self.model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0,
            in_chans=cfg.in_chans)

        self.n_features = self.model.num_features

        self.target_size = cfg.target_size if target_size is None else target_size

        # nn.Dropout(0.5),
        self.fc = nn.Sequential(
            nn.Linear(self.n_features, self.target_size)
        )

    def feature(self, image):

        feature = self.model(image)
        return feature

    def forward(self, image):
        feature = self.feature(image)
        output = self.fc(feature)
        return output

class CustomMetaModel(nn.Module):
    def __init__(self, cfg, pretrained=False, target_size=None, model_name=None):
        super().__init__()
        # self.cfg = cfg

        if model_name is None:
            model_name = cfg.model_name

        self.model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0,
            in_chans=cfg.in_chans)

        self.n_features = self.model.num_features

        self.target_size = cfg.target_size if target_size is None else target_size

        # nn.Dropout(0.5),
        """
        self.fc = nn.Sequential(
            nn.Linear(self.n_features, self.target_size)
        )
        """
        hidden_size = 64
        self.fc = nn.Sequential(
            nn.Linear(self.n_features +
                      len(cfg.meta_cols), hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            # nn.Linear(hidden_size, hidden_size),
            # nn.BatchNorm1d(hidden_size),
            # nn.ReLU(),
            nn.Linear(hidden_size, self.target_size)
        )

    def feature(self, image):

        feature = self.model(image)
        return feature

    def forward(self, image, meta_feature):
        feature = self.feature(image)
        feature = torch.cat([feature, meta_feature], dim=1)
        # print(feature.shape)
        output = self.fc(feature)
        return output
