import torch
import torch.nn as nn

class DrivingCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # CNN layers following PilotNet architecture
        self.cnn = nn.Sequential(
            # Input: 66x200x3 (resized from 768x1366x3)
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Fully connected layers for control outputs
        self.fc = nn.Sequential(
            nn.Linear(1152, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
            nn.ReLU(),
            nn.Linear(10, 4),  # 4 outputs: [throttle, brake, left, right]
            nn.Sigmoid()       # Output between 0 and 1 for each control
        )
        
    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)
        return x 