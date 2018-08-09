import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision import transforms, datasets
from PIL import Image, ImageDraw, ImageFont
import networks

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'
dataset = 'tiny_words'
#dataset = 'tiny_imagenet'

# Hyper parameters
num_epochs = 3
batch_size = 20
learning_rate = 0.001

def select_channel(X):
    """Our images are grayscale. Just select the first color channel."""
    return X[[0]]

def to_grayscale(X):
    """Convert image to grayscale"""
    return X.mean(dim=0).unsqueeze(0)

if dataset == 'tiny_words':
    image_size = 64
    num_channels = 1
    path = '/l/vanvlm1/word_stimuli'
    init = transforms.Compose([transforms.ToTensor(), select_channel])
elif dataset == 'tiny_imagenet':
    image_size = 64
    num_channels = 3
    path = '/l/vanvlm1/tiny-imagenet'
    init = transforms.Compose([
        transforms.RandomResizedCrop(int(image_size // 1.14)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
elif dataset == 'large_words':
    image_size = 128
    num_channels = 1
    path = '/l/vanvlm1/large_word_stimuli'
    init = transforms.Compose([transforms.ToTensor(), select_channel])

train_dataset = datasets.ImageFolder(root=path + '/train', transform=init)
val_dataset = datasets.ImageFolder(root=path + '/val', transform=init)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
)

val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, num_workers=4
)

num_classes = len(train_dataset.classes)

def accuracy(y_hat, y):
    with torch.no_grad():
        acc = torch.mean((y_hat.argmax(dim=1) == y).to(torch.float32))
    return acc.item()

model = networks.VGG16(image_size, num_channels, num_classes).to(device)

criterion = torch.nn.CrossEntropyLoss()
#optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum=0.9, weight_decay=1E-4)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1E-4)

# Train on the training set
for epoch in range(num_epochs):
    num_batches = len(train_dataloader)
    for i, (X, y) in enumerate(train_dataloader):
        X = X.to(device)
        y = y.to(device)

        y_hat = model(X)
        loss = criterion(y_hat, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print('Epoch: %d/%d  Train batch: %04d/%d  LR: %f  Loss: %.4f  Acc: %.4f' %
                  (epoch + 1, num_epochs, i, num_batches, learning_rate, loss.item(), accuracy(y_hat, y)))
    learning_rate /= 10
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Test on the validation set
num_batches = len(val_dataloader)
y_hat_val = np.zeros((num_batches, batch_size, num_classes))
y_val = np.zeros((num_batches, batch_size))
for i, (X, y) in enumerate(val_dataloader):
    y_hat = model(X.to(device))
    y_hat_val[i] = y_hat.cpu().detach().numpy()
    y_val[i] = y.cpu().detach().numpy()
    #print('Validation batch %03d/%d  Loss: %.4f  Acc: %.4f' % (i, num_batches, loss.item(), accuracy(y_hat, y.to(device))))
y_hat_val = y_hat_val.reshape(-1, num_classes)
y_val = y_val.ravel()
print('Final score:', np.mean(y_hat_val.argmax(axis=1) == y_val))

modulelist = list(model.features.modules())[1:] + list(model.classifier.modules())
def dd_helper(layer, iterations, lr, img=None):
    if img is None:
        input = torch.autograd.Variable(torch.rand(1, 1, 64, 64).cuda(), requires_grad=True)
    else:
        input = torch.autograd.Variable(img.unsqueeze(0).cuda(), requires_grad=True)
    model.zero_grad()
    for i in range(iterations):
#         print('Iteration: ', i)
        out = input
        for j in range(layer):
            out = modulelist[j](out)
        loss = out.norm()
        loss.backward()

        input.data = input.data + lr * input.grad.data
    return input.cpu().detach().numpy()[0]

def vis_layer(layer, img, num_cols=8):
    if type(img) == int:
        X, y = train_dataset[img]
    else:
        X = img
    plt.figure(1)
    plt.clf()
    plt.imshow(X[0])

    # Run the image through the layers
    X2 = X.unsqueeze(0).to(device)
    for i in range(layer + 1):
        X2 = modulelist[i](X2)
    X2 = X2.detach().cpu().numpy()

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
    return torch.tensor(np.asarray(canvas)[None, :, :, 0] / 255.0, dtype=torch.float32).to(device)
