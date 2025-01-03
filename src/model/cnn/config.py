"""Configuration for CNN model following PilotNet architecture"""

# Model Architecture
MODEL_CONFIG = {
    'input_channels': 15,  # 5 frames * 3 channels (YUV)
    'input_height': 66,    # PilotNet input size
    'input_width': 200,
    'conv_layers': [
        # (out_channels, kernel_size, stride)
        (24, 5, 2),
        (36, 5, 2),
        (48, 5, 2),
        (64, 3, 1),
        (64, 3, 1),
    ],
    'fc_layers': [1164, 100, 50, 10, 3],  # Following paper architecture
}

# Training Configuration (as per paper)
TRAINING_CONFIG = {
    'learning_rate': 1e-4,      # 0.0001 as mentioned in paper
    'weight_decay': 0.0,        # No L2 regularization mentioned
    'batch_size': 256,          # Large batch size for stable training
    'num_epochs': 100,
    'optimizer': 'Adam',        # Adam optimizer as per paper
    
    # Loss weights (our addition since we're predicting 3 outputs)
    'loss_weights': {
        'steering': 1.0,
        'throttle': 0.5,
        'brake': 0.5
    },
    
    # Data augmentation (as described in paper)
    'augmentation': {
        'enable': True,
        'shift_range': [-2, 2],      # Pixels
        'rotation_range': [-15, 15],  # Degrees
    }
}

# Input normalization
NORMALIZATION = {
    'yuv_mean': [0.5, 0, 0],      # Following YUV color space properties
    'yuv_std': [0.5, 1, 1],
} 