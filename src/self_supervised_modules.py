import torch.nn as nn
from torch import Tensor

"""
Encoder used for self-supervised algorithms
It's a typical encoder for 3D data
"""

class Encoder(nn.Module):
    def __init__(self, channels=None, out_size=128, image_planes=10, images_width=32, images_height=32):
        super().__init__()
        if channels is None:
            channels = [1, 16, 32, 64]
        
        self.out_size = out_size
        
        self.model = nn.Sequential()
        
        for i in range(len(channels)-1):
            self.model.add_module(f'conv_{i}', nn.Conv3d(channels[i], channels[i+1], kernel_size=3, padding=1))
            self.model.add_module(f'relu_{i}', nn.ReLU())
            self.model.add_module(f'maxpool_{i}', nn.MaxPool3d(2))

            image_planes = (image_planes - 2) // 2 + 1
            images_width = (images_width - 2) // 2 + 1
            images_height = (images_height - 2) // 2 + 1
            
        self.model.add_module("flatten", nn.Flatten())
        self.model.add_module("linear", nn.Linear(channels[-1]*image_planes*images_width*images_height, out_size))
    
    def forward(self, x):
        return self.model(x)
    
"""
Multilayer Perceptron implementation for BYOL
"""
    
class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, plain_last: bool = False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )
        if not plain_last:
            self.net.append(nn.BatchNorm1d(output_dim))
            self.net.append(nn.ReLU(inplace=True))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

"""
Classifier module used to classify data into nodules vs non-nodules
"""

class SelfSupervisedClassifier(nn.Module):
    def __init__(self, encoder, no_classes=2, freeze_encoder=True):
        super(SelfSupervisedClassifier, self).__init__()
        
        if freeze_encoder:
            for param in encoder.parameters():
                param.requires_grad = False

        self.encoder = encoder
        self.linear1 = nn.Linear(encoder.out_size, encoder.out_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.linear2 = nn.Linear(encoder.out_size, no_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.linear2(x)
