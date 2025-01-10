# models/pretrainedmodel.py

import torch.nn as nn
from torchvision import models

class PretrainedModel(nn.Module):
    def __init__(self, num_classes=10, use_pretrained=True):
        """
        Initialize the Pretrained Model based on ResNet18.

        Args:
            num_classes (int): Number of output classes.
            use_pretrained (bool): Whether to use pretrained weights.
        """
        super(PretrainedModel, self).__init__()
        # Load pre-trained ResNet18
        self.model = models.resnet18(pretrained=use_pretrained)
        
        # Modify the final layer for our number of classes
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        """
        Forward pass of the Pretrained Model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output logits.
        """
        return self.model(x)
