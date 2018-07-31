import torch
from torchvision import transforms, datasets
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
import numpy as np

import networks

model = networks.vgg(num_classes=200)
model.features = torch.nn.DataParallel(model.features)
checkpoint = torch.load('checkpoint.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
modulelist = list(model.features.modules())[2:] + list(model.classifier.modules())[1:]
model = model.cpu()

dataset = datasets.ImageFolder(
    '/l/vanvlm1/tiny_word_image/val',
    transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(60),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
)


def vis_layer(layer, img, num_cols=8):
    if type(img) == int:
        X, y = dataset[img]
    else:
        X = img
    plt.figure(1)
    plt.clf()
    plt.imshow(X[0])

    # Run the image through the layers
    X2 = X.unsqueeze(0)
    for i in range(layer + 1):
        X2 = modulelist[i](X2)
    X2 = X2.cpu().detach().numpy()

    # Visualize the result
    plot_kernels(X2[0], num_cols)

def vis_weights(layer):
    w = modulelist[layer].weight
    plt.figure(3)
    plt.clf()
    plt.imshow(w[:, 0, :, :].detach().reshape(w.shape[0] * w.shape[2], w.shape[3]), cmap='gray_r')
    plt.yticks(np.arange(0, w.shape[0] * w.shape[2], w.shape[2]))
    plt.grid(axis='y')
    plt.colorbar()

def plot_kernels(tensor, num_cols=8):
    if not tensor.ndim==3:
        raise Exception("assumes a 3D tensor")
    num_kernels = tensor.shape[0]
    num_rows = 1+ num_kernels // num_cols
    fig = plt.figure(2, figsize=(num_cols,num_rows))
    plt.clf()
    cmax = max(abs(tensor.min()), abs(tensor.max()))
    cmin = -cmax

    for i in range(tensor.shape[0]):
        ax1 = fig.add_subplot(num_rows,num_cols,i+1)
        im = ax1.imshow(tensor[i], cmap='RdBu_r')
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        im.set_clim(cmin, cmax)

    plt.subplots_adjust(wspace=0.1, hspace=0.1)


def text_phantom(text, size, fontsize=14, font='arial'):
    # Create font
    pil_font = ImageFont.truetype(
        font + ".ttf", size=fontsize, encoding="unic")
    text_width, text_height = pil_font.getsize(text)

    # create a blank canvas with extra space between lines
    canvas = Image.new('RGB', [size, size], (255, 255, 255))

    # draw the text onto the canvas
    draw = ImageDraw.Draw(canvas)
    offset = ((size - text_width) // 2,
              (size - text_height) // 2)
    draw.text(offset, text, font=pil_font, fill="#000000")

    # Convert the canvas into an array with values in [0, 1]
    return torch.tensor(np.asarray(canvas) / 255.0, dtype=torch.float32).transpose((2, 1, 0)).unsqueeze(0)
