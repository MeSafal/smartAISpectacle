# import torch.nn as nn
# from torchvision.models import resnet18
# from torchvision.models.resnet import ResNet18_Weights

# class MoneyClassifier(nn.Module):
#     def __init__(self, num_classes):
#         super(MoneyClassifier, self).__init__()
#         # Use a pre-trained ResNet model
#         self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
#         # Replace the last fully connected layer
#         num_features = self.resnet.fc.in_features
#         self.resnet.fc = nn.Linear(num_features, num_classes)

#     def forward(self, x):
#         return self.resnet(x)
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()  # Ensure the input is 4D
        se = self.avg_pool(x).view(b, c)  # Global average pooling
        se = self.fc(se).view(b, c, 1, 1)  # Squeeze and excitation
        return x * se  # Scale the input features

class MoneyClassifier(nn.Module):
    def __init__(self, num_classes):
        super(MoneyClassifier, self).__init__()
        # Use a pre-trained ResNet18 model
        self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # Adding a SE Block after a convolutional layer
        self.se_block = SEBlock(self.resnet.layer4[1].conv2.out_channels)

        # Replace the last fully connected layer with ReLU activation
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 512),  # Add a hidden layer
            nn.ReLU(),                     # Activation
            nn.Linear(512, num_classes)    # Final classification layer
        )

    def forward(self, x):
        # Extract features before the fully connected layers
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x) 
        
        # Apply SEBlock to the feature maps
        x = self.se_block(x)

        # Global average pooling
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Pass through the fully connected layers
        x = self.resnet.fc(x)
        return x
