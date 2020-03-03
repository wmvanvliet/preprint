import torch
from torch import nn

class TwoLayerNet(nn.Module):
    def __init__(self, image_size=64, num_channels=3, num_classes=200):
        super().__init__()

        n_filters1 = 16
        n_filters2 = 32
        self.features = nn.Sequential(
            nn.Conv2d(num_channels, n_filters1, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(n_filters1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(n_filters1, n_filters2, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(n_filters2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.classifier = nn.Linear(n_filters2 * 12 * 12, num_classes)
        
    def forward(self, X):
        out = self.features(X)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


class ThreeLayerNet(nn.Module):
    def __init__(self, image_size=64, num_channels=3, num_classes=200):
        super().__init__()

        n_filters1 = 16
        n_filters2 = 16
        n_filters3 = 16
        self.features = nn.Sequential(
            nn.Conv2d(num_channels, n_filters1, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(n_filters1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_filters1, n_filters2, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(n_filters2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(n_filters2, n_filters3, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(n_filters3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))

        if image_size == 64:
            self.classifier = nn.Linear(n_filters2 * 14 * 14, num_classes)
        elif image_size == 128:
            self.classifier = nn.Linear(n_filters2 * 30 * 30, num_classes)

    def forward(self, X):
        out = self.features(X)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
