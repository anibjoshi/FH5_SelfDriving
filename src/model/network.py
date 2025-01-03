import torch
import torch.nn as nn
from src.config import INPUT_HEIGHT, INPUT_WIDTH

class DrivingCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # CNN layers following PilotNet architecture
        self.cnn = nn.Sequential(
            # Input: INPUT_HEIGHTxINPUT_WIDTHx3
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ReLU(),
            # ... rest of the implementation ...
        ) 