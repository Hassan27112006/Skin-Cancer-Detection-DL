import torch.nn as nn
from torchvision import models

def get_model(name, num_classes):
    if name == "efficientnet":
        model = models.efficientnet_b0(pretrained=True)
        model.classifier[1] = nn.Linear(1280, num_classes)

    elif name == "resnet":
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(2048, num_classes)

    return model
