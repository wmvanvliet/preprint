import numpy as np
import torch
from torch import nn
from torchvision import datasets, transforms


def gabor_filter(size, orientation, wavelength=2):
    """Create a single gabor filter.

    Parameters
    ----------
    size : int
        The size of the filter, measured in pixels. The filter is square, hence
        only a single number (either width or height) needs to be specified.
    orientation : float
        The orientation of the grating in the filter, in degrees.
    wavelength : float (default: 2)
        The wavelength of the grating in the filter, relative to the half the
        size of the filter. For example, a wavelength of 2 will generate a
        Gabor filter with a grating that contains exactly one wave. This
        determines the "tightness" of the filter.

    Returns
    -------
    filt : ndarray, shape (size, size)
        The filter weights.
    """
    lambda_ = size * 2. / wavelength
    sigma = lambda_ * 0.8
    gamma = 0.3  # spatial aspect ratio: 0.23 < gamma < 0.92
    theta = np.deg2rad(orientation + 90)

    # Generate Gabor filter
    x, y = np.mgrid[:size, :size] - (size // 2)
    rotx = x * np.cos(theta) + y * np.sin(theta)
    roty = -x * np.sin(theta) + y * np.cos(theta)
    filt = np.exp(-(rotx**2 + gamma**2 * roty**2) / (2 * sigma ** 2))
    filt *= np.cos(2 * np.pi * rotx / lambda_)
    filt[np.sqrt(x**2 + y**2) > (size / 2)] = 0

    # Normalize the filter
    filt = filt - np.mean(filt)
    filt = filt / np.sqrt(np.sum(filt ** 2))

    return filt


class S1(nn.Module):
    """A module that normalizes the result with the local norm of the image"""
    def __init__(self, size=7, wavelength=4):
        super().__init__()

        # 4 Gabor filters in different orientations
        self.gabor = nn.Conv2d(1, 4, size, padding=size // 2, bias=False)
        for i, orientation in enumerate([90, -45, 0, 45]):
            self.gabor.weight.data[i, 0] = torch.Tensor(
                gabor_filter(size, orientation, wavelength))

        # A convolution layer filled with ones. This is used to normalize the
        # result in the forward method.
        self.uniform = nn.Conv2d(1, 4, size, padding=size // 2, bias=False)
        nn.init.constant_(self.uniform.weight, 1)

    def forward(self, X):
        """Apply Gabor filters, take absolute value, and normalize"""
        out = torch.abs(self.gabor(X))
        norm = torch.sqrt(self.uniform(X ** 2))
        norm.data[norm == 0] = 1  # To avoid divide by zero
        out /= norm
        return out


class MaxPool(nn.Module):
    """A MaxPool2D that also pools across channels."""
    def __init__(self, c1_space=8):
        super().__init__()

        # Pools over all channels (i.e. Gabor orientations) and pixels within a
        # local sliding window.
        self.pool = nn.MaxPool3d((4, c1_space, c1_space))

    def forward(self, X):
        # Add a dummy dimension and remove it again to make MaxPool3d work
        return self.pool(X[:, None, :, :])[:, 0, :, :]


class HMAX(nn.Module):
    def __init__(self, size=7, wavelength=4, c1_space=8):
        super().__init__()

        self.s1 = S1(size, wavelength)
        self.c1 = MaxPool(c1_space)

    def forward(self, X):
        X = self.s1(X)
        X = self.c1(X)
        return X


example_images = datasets.ImageFolder(
    '/l/vanvlm1/example_images',
    transform=transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255),
    ])
)

model = HMAX()
