import torch
import torch.nn as nn

print("=== 03. Vision Concept (CNNs) ===")

# Simulated Image Data: [Batch, Channels, Height, Width]
# example: 4 color images of 32x32 pixels
images = torch.randn(4, 3, 32, 32)
print(f"Input image batch shape: {images.shape}")

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Conv2d(in_channels, out_channels, kernel_size)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2) # Halves H and W
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * 8 * 8, 10) # 8x8 is result after 2 poolings

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8) # Flatten
        x = self.fc(x)
        return x

model = SimpleCNN()
output = model(images)
print(f"Output shape (4 images, 10 classes): {output.shape}\n")
