import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import pickle
import cv2

# It provides heat map to show which part of imaage is focused during prediction or decision making

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook) 

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()

        output = self.model(input_tensor)
        loss = output[:, class_idx]
        loss.backward()

        grads = self.gradients
        activations = self.activations


        weights = torch.mean(grads, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations, dim=1)

        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()


        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        return cam