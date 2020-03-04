import torch
from torch import nn


class VGG16(nn.Module):
    def __init__(self, num_channels=3, num_classes=200, classifier_size=4096):
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

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 3 * 3, classifier_size),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(classifier_size, classifier_size),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(classifier_size, num_classes),
            nn.ReLU(True),
            nn.LogSoftmax(),
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

    @classmethod
    def from_checkpoint(cls, checkpoint, num_classes=None, freeze=False):
        state_dict = checkpoint['state_dict']
        num_channels = state_dict['features.0.weight'].shape[1]
        prev_num_classes = state_dict['classifier.6.weight'].shape[0]
        classifier_size = state_dict['classifier.6.weight'].shape[1]

        model = cls(num_channels, prev_num_classes, classifier_size)
        model.load_state_dict(state_dict)

        if num_classes is not None:
            if freeze:
                print('=> freezing model')
                for param in model.parameters():
                    param.requires_grad = False

            print(f'=> attaching new output layer (size changed from {prev_num_classes} to {num_classes})')
            modulelist = list(model.classifier.modules())[1:]
            modulelist.pop()
            classifier3 = nn.Linear(classifier_size, num_classes)
            nn.init.normal_(classifier3.weight, 0, 0.01)
            nn.init.constant_(classifier3.bias, 0)
            modulelist.append(classifier3)
            model.classifier = nn.Sequential(*modulelist)

        return model


class VGGSem(nn.Module):
    @classmethod
    def from_checkpoint(cls, checkpoint, num_classes=300, freeze=True):
        state_dict = checkpoint['state_dict']
        num_channels = state_dict['features.0.weight'].shape[1]
        prev_num_classes = state_dict['classifier.6.weight'].shape[0]
        classifier_size = state_dict['classifier.6.weight'].shape[1]

        vis_model = VGG16(num_channels, prev_num_classes, classifier_size)
        vis_model.load_state_dict(state_dict)

        if num_classes is not None:
            if freeze:
                print('=> freezing visual model')
                for param in vis_model.parameters():
                    param.requires_grad = False

        print(f'=> attaching semantic layer (going from {prev_num_classes} to {num_classes})')
        return cls(vis_model, num_classes)

    def __init__(self, vis_network, num_classes=300, classifier_size=1024):
        super().__init__()
        self.vis = vis_network
        num_words = vis_network.classifier[-1].weight.shape[0]

        # Stack on some semantic layers
        self.semantics = nn.Sequential(
            nn.Linear(num_words, classifier_size),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(classifier_size, num_classes),
            nn.ReLU(True),
            nn.LogSoftmax(),
        )
        #self.semantics = nn.Linear(num_words, num_classes)
        #nn.init.normal_(self.semantics.weight, 0, 0.01)
        #nn.init.constant_(self.semantics.bias, 0)
        self.initialize_semantic_weights()

    def initialize_semantic_weights(self):
        for m in self.semantics.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, X):
        out = self.vis(X)
        out = self.semantics(out)
        return out
