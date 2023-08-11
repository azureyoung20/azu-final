import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# Load the image
image = Image.open('F:/PATCHES_NORMAL_VALID/108.png')
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to 224x224 pixels
    transforms.ToTensor(),  # Convert the image to PyTorch Tensor
])
image = transform(image)
image = image.unsqueeze(0)  # Add an extra dimension for batch size

# Define a single convolutional layer
conv_layer = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1)

# Apply the convolutional layer to the image
feature_maps = conv_layer(image)

# Visualize the feature maps
fig, axs = plt.subplots(1, 6, figsize=(20, 20))
for i, feature_map in enumerate(feature_maps[0]):
    axs[i].imshow(feature_map.detach().numpy())
plt.show()
