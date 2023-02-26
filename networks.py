import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
class Resnet(nn.Module):
    def __init__(self, model_name, n_class=2, pretrained=True):
        super().__init__()
        print(model_name)
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(n_features, n_class)

    def forward(self, x):
        x = self.model(x)
        return x