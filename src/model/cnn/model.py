import torch
import torch.nn as nn

class CNNModel(nn.Module):
    """PilotNet architecture from NVIDIA paper"""
    def __init__(self):
        super().__init__()
        
        # CNN layers (5 stacked RGB frames, YUV color space)
        self.cnn = nn.Sequential(
            nn.Conv2d(15, 24, kernel_size=5, stride=2),
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
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(1152 + 5, 1164),  # 1152 from CNN + 5 telemetry values
            nn.ReLU(),
            nn.Linear(1164, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
            nn.ReLU(),
            nn.Linear(10, 3)  # [steering, throttle, brake]
        )
        
        self.steering_activation = nn.Tanh()    # [-1, 1]
        self.pedal_activation = nn.Sigmoid()    # [0, 1]
    
    def forward(self, frames, telemetry):
        visual_features = self.cnn(frames)
        features = torch.cat([visual_features, telemetry], dim=1)
        outputs = self.fc(features)
        
        controls = torch.zeros_like(outputs)
        controls[:, 0] = self.steering_activation(outputs[:, 0])
        controls[:, 1] = self.pedal_activation(outputs[:, 1])
        controls[:, 2] = self.pedal_activation(outputs[:, 2])
        
        return controls 