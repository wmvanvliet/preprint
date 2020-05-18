import torch
from torch import nn


class WinnerTakesAll(nn.Module):
    def forward(self, x):
        out = torch.zeros_like(x)
        out[(torch.arange(len(out)), torch.argmax(x, dim=1))] = 1
        return out


class VGG16(nn.Module):
    """VGG-16 model as described in the literature. With some extra convenience methods."""
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
            #nn.ReLU(True),  # <-- Added after training
            #WinnerTakesAll(),  # <-- Added after training
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

    def get_layer_activations(self, images):
        """Obtain activation of each layer on a set of images."""
        self.eval()
        batch_size = 128
        feature_outputs = []
        out = images
        for i, layer in enumerate(self.features):
            layer_out = []
            for j in range(0, len(out), batch_size):
                layer_out.append(layer(out[j:j + batch_size]))
            out = torch.cat(layer_out, 0)
            del layer_out
            print('feature layer %02d, output=%s' % (i, out.shape))
            #if i in [3, 10, 17, 24]:
            if i in [5, 12, 19, 26]:
                feature_outputs.append(out.detach().numpy().copy())
        classifier_outputs = []
        out = out.view(out.size(0), -1)
        layer_out = []
        for i, layer in enumerate(self.classifier):
            layer_out = []
            for j in range(0, len(out), batch_size):
                layer_out.append(layer(out[j:j + batch_size]))
            out = torch.cat(layer_out, 0)
            print('classifier layer %02d, output=%s' % (i, out.shape))
            #if i in [0, 3, 6]:
            if i in [1, 4, 8]:
                classifier_outputs.append(out.detach().numpy().copy())
        return feature_outputs, classifier_outputs

    def set_n_outputs(self, num_classes, keep_weights=True):
        modulelist = list(self.classifier.modules())[1:]
        output_layer = modulelist[6]
        weight = output_layer.weight[:num_classes, :]
        bias = output_layer.bias[:num_classes]
        prev_num_classes, classifier_size = weight.shape
        print(f'=> resizing output layer ({prev_num_classes} => {num_classes}), keep_weights={keep_weights}')
        output_layer = nn.Linear(classifier_size, num_classes)
        if keep_weights:
            output_layer.weight = nn.Parameter(weight)
            output_layer.bias = nn.Parameter(bias)
        else:
            nn.init.normal_(output_layer.weight, 0, 0.01)
            nn.init.constant_(output_layer.bias, 0)
        modulelist[6] = output_layer
        self.classifier = nn.Sequential(*modulelist)

    @classmethod
    def from_checkpoint(cls, checkpoint, num_classes=None, freeze=False):
        """Construct this model from a stored checkpoint."""
        state_dict = checkpoint['state_dict']
        num_channels = state_dict['features.0.weight'].shape[1]
        prev_num_classes = state_dict['classifier.6.weight'].shape[0]
        classifier_size = state_dict['classifier.6.weight'].shape[1]

        model = cls(num_channels, prev_num_classes, classifier_size)
        model.load_state_dict(state_dict)

        if num_classes is not None:
            if freeze:
                print('=> freezing model')
                for layer in model.features[:14]:
                    for param in layer.parameters():
                        param.requires_grad = False
                #print('=> disabling tracking batchnorm running stats')
                #for m in model.modules():
                #    if isinstance(m, nn.BatchNorm2d):
                #        m.track_running_stats = False

            model.set_n_outputs(num_classes)

        return model


class VGGSem(nn.Module):
    """Model that wraps a VGG16 model and appends some semantic layers."""
    @classmethod
    def from_checkpoint(cls, checkpoint, num_classes=None, vector_length=300, freeze=False):
        """Construct this model from a stored checkpoint of a VGG16 model."""
        state_dict = checkpoint['state_dict']
        num_channels = state_dict['features.0.weight'].shape[1]
        prev_num_classes = state_dict['classifier.6.weight'].shape[0]
        classifier_size = state_dict['classifier.6.weight'].shape[1]
        vis_model = VGG16(num_channels, prev_num_classes, classifier_size)
        
        if checkpoint['arch'] == 'vgg':
            vis_model.load_state_dict(state_dict)

            if num_classes is not None:
                vis_model.set_n_outputs(num_classes)
            else:
                num_classes = prev_num_classes

            if freeze:
                print('=> freezing feature and classifier parts of the model')
                for param in vis_model.parameters():
                    param.requires_grad = False
                print('=> disabling tracking batchnorm running stats')
                for m in vis_model.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.track_running_stats = False
            print(f'=> attaching semantic layer (going from {prev_num_classes} to {num_classes})')
            model = cls(vis_model, vector_length)
        elif checkpoint['arch'] == 'vgg_sem':
            vector_length = state_dict['semantics.0.weight'].shape[0]
            model = VGGSem(vis_model, vector_length)
            model.load_state_dict(state_dict)
            if freeze:
                print('=> freezing all parts of the model')
                for param in model.parameters():
                    param.requires_grad = False
                print('=> disabling tracking batchnorm running stats')
                for m in model.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.track_running_stats = False
        else:
            raise ValueError(f"Invalid network architecture in checkpoint: {checkpoint['arch']}")
        model.eval()
        return model

    def __init__(self, vis_network, vector_size=300):
        super().__init__()
        self.features = vis_network.features
        self.classifier = vis_network.classifier
        num_classes = self.classifier[6].weight.shape[0]

        # Stack on some semantic layers
        self.semantics = nn.Sequential(
            nn.Linear(num_classes, vector_size, bias=False),
        )
        self.initialize_semantic_weights()

    def initialize_semantic_weights(self):
        for m in self.semantics.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, X):
        out = self.features(X)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        out = self.semantics(out)
        return out

    def get_layer_activations(self, images):
        """Obtain activation of each layer on a set of images."""
        self.eval()
        batch_size = 128
        feature_outputs = []
        out = images
        for i, layer in enumerate(self.features):
            layer_out = []
            for j in range(0, len(out), batch_size):
                layer_out.append(layer(out[j:j + batch_size]))
            out = torch.cat(layer_out, 0)
            del layer_out
            print('feature layer %02d, output=%s' % (i, out.shape))
            #if i in [5, 12, 19, 26]:
            if i in [3, 10, 17, 24]:
                feature_outputs.append(out.detach().numpy().copy())
        classifier_outputs = []
        out = out.view(out.size(0), -1)
        layer_out = []
        for i, layer in enumerate(self.classifier):
            layer_out = []
            for j in range(0, len(out), batch_size):
                layer_out.append(layer(out[j:j + batch_size]))
            out = torch.cat(layer_out, 0)
            print('classifier layer %02d, output=%s' % (i, out.shape))
            #if i in [1, 4, 8]:
            if i in [0, 3, 6]:
                classifier_outputs.append(out.detach().numpy().copy())
        semantic_outputs = []
        for i, layer in enumerate(self.semantics):
            layer_out = []
            for j in range(0, len(out), batch_size):
                layer_out.append(layer(out[j:j + batch_size]))
            out = torch.cat(layer_out, 0)
            print('semantic layer %02d, output=%s' % (i, out.shape))
            if i in [0]:
                semantic_outputs.append(out.detach().numpy().copy())
        return feature_outputs, classifier_outputs, semantic_outputs
