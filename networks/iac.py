import torch
from torch import nn, cat


class Combine(nn.Module):
    def __init__(self, left, right):
        super().__init__()
        self.left = left
        self.right = right

    def forward(self, X):
        return cat((self.left(X), self.right(X)), dim=1)


class IAC(nn.Module):
    """A pool of interactive activation and competition units."""
    def __init__(self, weights, max_activation=1, min_activation=-0.2, rest_activation=-0.1, decay=0.1, gamma=0.1):
        super().__init__()
        input_size, output_size = weights.shape
        self.weight = nn.Parameter(weights)
        self.activation = rest_activation * torch.ones(1, output_size)
        self.activation.requires_grad = False
        self.min_activation = min_activation
        self.max_activation = max_activation
        self.rest_activation = rest_activation
        self.decay = decay
        self.gamma = gamma

    def reset(self):
        self.activation = self.rest_activation * torch.ones(1, self.weights.shape[0])

    def forward(self, X):
        #print('X= ', X)
        #print('W= ', self.weight)
        excitation = torch.matmul(X, self.weight)  # Excitation
        #print('E= ', excitation)
        inhibition = torch.sum(torch.clamp_min(self.activation, 0)) - torch.clamp_min(self.activation, 0)  # Inhibition
        #print('I= ', inhibition)
        net = excitation - self.gamma * inhibition
        #print('N= ', net)
        #print('A= ', self.activation)
        net[net > 0] *= (self.max_activation - self.activation)[net > 0]
        net[net <= 0] *= (self.activation - self.min_activation)[net <= 0]
        net -= self.decay * (self.activation - self.rest_activation)
        #print('N= ', net)
        self.activation += net
        print('A= ', self.activation)
        return torch.clamp_min(self.activation, 0)
##

class IACConv(nn.Module):
    """Convolution layer implemented using IAC unites."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros', max_activation=1, min_activation=-0.2,
                 rest_activation=-0.1, decay=0.1, alpha=0.1, gamma=0.1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=False,
                              padding_mode=padding_mode)
        self.inhibition = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2,
                      padding=0, bias=False),
            nn.Upsample(scale_factor=2),
        )
        nn.init.constant(self.inhibition[0].weight, 1)
        self.inhibition[0].required_grad = False
        self.activation = None
        self.min_activation = min_activation
        self.max_activation = max_activation
        self.rest_activation = rest_activation
        self.decay = decay
        self.alpha = alpha
        self.gamma = gamma

    def reset(self):
        if self.activation is not None:
            self.activation = nn.init.constant_(self.activation, self.rest_activation)

    def forward(self, X):
        excitation = self.conv(X)
        self.excitation = excitation
        if self.activation is None:
            self.activation = self.rest_activation * torch.ones(*excitation.shape)
        inhibition = self.inhibition(torch.clamp_min(self.activation, 0)) - torch.clamp_min(self.activation, 0)  # Inhibition
        self.inh = inhibition
        net = self.alpha * excitation - self.gamma * inhibition
        net[net > 0] *= (self.max_activation - self.activation)[net > 0]
        net[net <= 0] *= (self.activation - self.min_activation)[net <= 0]
        self.net1 = torch.tensor(net)
        net -= self.decay * (self.activation - self.rest_activation)
        self.net2 = torch.tensor(net)
        self.activation += net
        torch.clamp_(self.activation, self.min_activation, self.max_activation)
        return torch.clamp_min(self.activation, 0)
