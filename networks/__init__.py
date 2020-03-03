from torchvision import models
from . import simple
from . import reading
from . import iac

twolayer = simple.TwoLayerNet
alexnet = models.alexnet
vgg16 = models.vgg16
resnet18 = models.resnet18
vgg = reading.VGG16
vgg_sem = reading.VGGSem
