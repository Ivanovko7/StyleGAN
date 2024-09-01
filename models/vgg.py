import torchvision.models as models
import torch.nn as nn
import torch

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
WEIGHTS = 'DEFAULT'

class Vgg19(nn.Module):
    def __init__(self):
        super(Vgg19, self).__init__()
        self.vgg19 = self.get_vgg19().eval()
        vgg_mean = torch.tensor(MEAN).float()
        vgg_std = torch.tensor(STD).float()
        self.mean = vgg_mean.view(-1, 1 ,1)
        self.std = vgg_std.view(-1, 1, 1)

    def to(self, device):
        new_self = super(Vgg19, self).to(device)
        new_self.mean = new_self.mean.to(device)
        new_self.std = new_self.std.to(device)
        return new_self

    def forward(self, x):
        return self.vgg19(self.normalize_vgg(x))

    @staticmethod
    def get_vgg19(last_layer='conv4_4'):
        vgg = models.vgg19(weights=WEIGHTS).features
        model_list = []

        i = 0
        j = 1
        for layer in vgg.children():
            if isinstance(layer, nn.MaxPool2d):
                i = 0
                j += 1

            elif isinstance(layer, nn.Conv2d):
                i += 1

            name = f'conv{j}_{i}'

            if name == last_layer:
                model_list.append(layer)
                break

            model_list.append(layer)


        model = nn.Sequential(*model_list)
        return model


    def normalize_vgg(self, image):
        '''
        Expect input in range -1 1
        '''
        image = (image + 1.0) / 2.0
        return (image - self.mean) / self.std

