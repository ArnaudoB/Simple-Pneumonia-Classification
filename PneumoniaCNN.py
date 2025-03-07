import torch.nn as nn

class PneumoniaCNN(nn.Module):
    def __init__(self):
        super(PneumoniaCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(32*56*56, 128)
        self.fc2 = nn.Linear(128, 2)  # 2 classes (NORMAL, PNEUMONIA)

        self.relu = nn.ReLU()
        
    def forward(self, x):

        x = self.conv1(x)      # [batch, 16, 224, 224]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)      # [batch, 16, 112, 112]

        x = self.conv2(x)      # [batch, 32, 112, 112]
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)      # [batch, 32, 56, 56]

        x = x.view(x.size(0), -1)  # [batch, 32 * 56 * 56]
        
        x = self.relu(self.fc1(x)) # [batch, 128]
        x = self.fc2(x)            # [batch, 2]
        
        return x