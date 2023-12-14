import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryClassifierCNN(nn.Module):
    def __init__(self):
        super(BinaryClassifierCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 256 * 256, 128)  # Adjust input size based on the output size of the last convolutional layer
        self.fc2 = nn.Linear(128, 2)  # Two output neurons for binary classification

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 256 * 256)  # Flatten for fully connected layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Example usage:
# Create an instance of the model
model = BinaryClassifierCNN()

# Print the model architecture
print(model)

# Dummy input with shape (batch_size, channels, height, width)
dummy_input = torch.randn(1, 3, 512, 512)

# Forward pass to check the output shape
output = model(dummy_input)
print("Output shape:", output.shape)