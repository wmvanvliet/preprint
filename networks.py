from torch import nn
from torchvision import models

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

        if image_size == 64:
            self.classifier = nn.Linear(n_filters2 * 13 * 13, num_classes)
        elif image_size == 128:
            self.classifier = nn.Linear(n_filters2 * 30 * 30, num_classes)
        
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


class VGG16(nn.Module):
    def __init__(self, image_size=64, num_channels=1, num_classes=200):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # nn.Conv2d(256, 512, kernel_size=3, padding=1),
            # nn.BatchNorm2d(512),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(512, 512, kernel_size=3, padding=1),
            # nn.BatchNorm2d(512),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2),
        )

        if image_size == 64:
            size = 8
        elif image_size == 128:
            size = 7

        self.classifier = nn.Sequential(
            nn.Linear(256 * size * size, 256),
            nn.ReLU(True),
            #nn.Dropout(),
            #nn.Linear(4096, 4096),
            #nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(256, num_classes),
        )

        self.initialize_weights()

    def forward(self, X):
        out = self.features(X)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

twolayer = TwoLayerNet
alexnet = models.alexnet
