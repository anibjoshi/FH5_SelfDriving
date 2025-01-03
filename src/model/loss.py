import torch
import torch.nn as nn
import torch.nn.functional as F

class DrivingLoss(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.weights = weights or {
            'steering': 1.0,
            'throttle': 0.5,
            'brake': 0.5
        }
    
    def forward(self, predictions, targets, metadata=None):
        """
        Compute weighted loss for driving controls
        Args:
            predictions: (B, 3) tensor [steering, throttle, brake]
            targets: (B, 3) tensor [steering, throttle, brake]
            metadata: Optional dict with target ranges
        """
        # Unpack predictions and targets
        pred_steering = predictions[:, 0]
        pred_throttle = predictions[:, 1]
        pred_brake = predictions[:, 2]
        
        target_steering = targets[:, 0]
        target_throttle = targets[:, 1]
        target_brake = targets[:, 2]
        
        # Compute individual losses
        steering_loss = F.mse_loss(pred_steering, target_steering)
        throttle_loss = F.mse_loss(pred_throttle, target_throttle)
        brake_loss = F.mse_loss(pred_brake, target_brake)
        
        # Apply weights
        total_loss = (
            self.weights['steering'] * steering_loss +
            self.weights['throttle'] * throttle_loss +
            self.weights['brake'] * brake_loss
        )
        
        # Return total loss and components
        return total_loss, {
            'total_loss': total_loss.item(),
            'steering_loss': steering_loss.item(),
            'throttle_loss': throttle_loss.item(),
            'brake_loss': brake_loss.item()
        } 