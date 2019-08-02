"""
Created on Thu Oct 26 11:23:47 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import torch
from torch.nn import ReLU
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from misc_functions import (get_params,
                            convert_to_grayscale,
                            save_gradient_images,
                            get_positive_negative_saliency,
                            preprocess_image)

import networks


class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        # Put model in evaluation mode
        self.model.eval()
        self.update_relus()
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]

        # Register hook to the first layer
        first_layer = list(self.model.features._modules.items())[0][1][0]
        first_layer.register_backward_hook(hook_function)

    def update_relus(self):
        """
            Updates relu activation functions so that it only returns positive gradients
        """
        def relu_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, changes it to zero
            """
            if isinstance(module, ReLU):
                return (torch.clamp(grad_in[0], min=0.0),)
        # Loop through layers, hook up ReLUs with relu_hook_function
        for pos, module in self.model.features._modules.items():
            if isinstance(module, ReLU):
                module.register_backward_hook(relu_hook_function)

    def generate_gradients(self, input_image, target_class):
        # Forward pass
        model_output = self.model(input_image)
        # Zero gradients
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Backward pass
        model_output.backward(gradient=one_hot_output.cuda())
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.cpu().data.numpy()[0]
        return gradients_as_arr

model = networks.vgg(num_classes=400)
model.features = torch.nn.DataParallel(model.features)
checkpoint = torch.load('models/vgg_first_images_then_words.pth.tar')
#checkpoint = torch.load('models/vgg_word_stimuli.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
model = model.eval()
model = model.cuda()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

dataset = datasets.ImageFolder(
    '/l/vanvlm1/word_stimuli/val',
    transforms.Compose([
        transforms.CenterCrop(60),
        transforms.ToTensor(),
        normalize,
    ]))

for i in range(256):
    prep_img, target_class = dataset[i]
    prep_img = prep_img.unsqueeze(0)
    prep_img = torch.autograd.Variable(prep_img, requires_grad=True)
    prefix = ('./generated/%03d' % i) + dataset.classes[target_class]

    # Guided backprop
    GBP = GuidedBackprop(model)
    # Get gradients
    guided_grads = GBP.generate_gradients(prep_img.cuda(), target_class)
    # Save colored gradients
    save_gradient_images(guided_grads, prefix + '_Guided_BP_color')
    # Convert to grayscale
    grayscale_guided_grads = convert_to_grayscale(guided_grads)
    # Save grayscale gradients
    save_gradient_images(grayscale_guided_grads, prefix + '_Guided_BP_gray')
    # Positive and negative saliency maps
    pos_sal, neg_sal = get_positive_negative_saliency(guided_grads)
    pos_sal = convert_to_grayscale(pos_sal)
    neg_sal = convert_to_grayscale(neg_sal)
    save_gradient_images(pos_sal, prefix + '_pos_sal')
    save_gradient_images(neg_sal, prefix + '_neg_sal')
    print('Guided backprop completed')
