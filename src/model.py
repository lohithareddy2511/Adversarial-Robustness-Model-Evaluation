"""
Base CNN model for CIFAR-10 classification with optional MC Dropout support.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Basic residual block with two conv layers."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class RobustCNN(nn.Module):
    """
    ResNet-style CNN for CIFAR-10.
    Supports MC Dropout for uncertainty estimation when mc_dropout=True.
    """

    def __init__(self, num_classes=10, dropout_rate=0.1, mc_dropout=False):
        super().__init__()
        self.mc_dropout = mc_dropout
        self.dropout_rate = dropout_rate

        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(64, 64, stride=1)
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)

        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, in_channels, out_channels, stride):
        return nn.Sequential(
            ResidualBlock(in_channels, out_channels, stride),
            ResidualBlock(out_channels, out_channels, 1),
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)

        if self.mc_dropout:
            # Always apply dropout (even at eval time) for MC Dropout
            out = F.dropout(out, p=self.dropout_rate, training=True)
        else:
            out = self.dropout(out)

        return self.fc(out)

    def enable_mc_dropout(self):
        """Enable MC Dropout mode for uncertainty estimation."""
        self.mc_dropout = True

    def disable_mc_dropout(self):
        """Disable MC Dropout mode."""
        self.mc_dropout = False
