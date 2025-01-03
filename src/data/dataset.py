import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from pathlib import Path
import numpy as np
import json
import os
from glob import glob

class DrivingDataset(Dataset):
    def __init__(self, data_dir: str):
        super().__init__()
        self.data_dir = Path(data_dir)
        
        # Load all samples
        self.samples = []
        self._load_samples()
        
        # PilotNet transform: resize to 66x200 and normalize
        self.transform = T.Compose([
            T.Resize((66, 200)),
            T.Normalize(mean=[0.5, 0, 0], std=[0.5, 1, 1])  # YUV normalization
        ])
    
    def _load_samples(self):
        """Load all valid samples from the processed data directory"""
        clip_dirs = glob(os.path.join(self.data_dir, "**/clip_*"), recursive=True)
        
        for clip_dir in clip_dirs:
            with open(os.path.join(clip_dir, "processed_data.json"), 'r') as f:
                clip_data = json.load(f)
                frames = [f for f in clip_data['frames'] 
                         if not f['padded'] and 'processed_file' in f]
                
                for frame in frames:
                    self.samples.append({
                        'frame_path': os.path.join(clip_dir, "processed", frame['processed_file']),
                        'telemetry': [
                            frame['telemetry']['speed'],
                            frame['telemetry']['yaw'],
                            frame['telemetry']['yaw_diff'],
                            frame['telemetry'].get('accel_x', 0.0),
                            frame['telemetry'].get('angular_velocity_y', 0.0)
                        ],
                        'targets': [
                            frame['telemetry']['steering'],
                            frame['telemetry']['throttle'],
                            frame['telemetry']['brake']
                        ]
                    })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load stacked frames
        frames = np.load(sample['frame_path'])
        frames = torch.from_numpy(frames).float() / 255.0
        
        # Apply transform
        frames = self.transform(frames)
        
        # Load telemetry and targets
        telemetry = torch.tensor(sample['telemetry'], dtype=torch.float32)
        targets = torch.tensor(sample['targets'], dtype=torch.float32)
        
        return {
            'frames': frames,
            'telemetry': telemetry,
            'targets': targets
        } 