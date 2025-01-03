import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os
from glob import glob
import torchvision.transforms as T
from PIL import Image
import random
import pandas as pd
from scikit_learn.model_selection import train_test_split
from torch.utils.data import DataLoader

class DrivingDataset(Dataset):
    def __init__(self, processed_dir, transform=True, cache_size=100):
        """
        Args:
            processed_dir: Directory containing processed clips
        """
        self.processed_dir = processed_dir
        self.samples = []
        
        # Load all clips
        clip_dirs = glob(os.path.join(processed_dir, "**/clip_*"), recursive=True)
        for clip_dir in clip_dirs:
            with open(os.path.join(clip_dir, "processed_data.json"), 'r') as f:
                clip_data = json.load(f)
                
                # Add all non-padded frames from clip
                frames = [f for f in clip_data['frames'] 
                         if not f['padded'] and 'processed_file' in f]
                self.samples.extend([
                    {
                        'clip_dir': clip_dir,
                        'frame_data': frame
                    } for frame in frames
                ])
        
        self.transform = transform
        
        # Compute telemetry stats for normalization
        self.telemetry_mean = np.zeros(5)
        self.telemetry_std = np.ones(5)
        if transform:
            self._compute_telemetry_stats()
            
        self.cache = {}
        self.cache_size = cache_size
        
        # Add value ranges for normalization
        self.target_ranges = {
            'steering': (-1.0, 1.0),
            'throttle': (0.0, 1.0),
            'brake': (0.0, 1.0)
        }
        
        self.telemetry_ranges = {
            'speed': (0.0, 100.0),  # m/s
            'yaw': (-180.0, 180.0),
            'yaw_diff': (-10.0, 10.0),
            'accel_x': (-30.0, 30.0),
            'angular_velocity_y': (-3.0, 3.0)
        }
        
    def _compute_telemetry_stats(self):
        """Compute mean and std of telemetry features for normalization"""
        print("Computing telemetry statistics...")
        features = []
        for sample in self.samples:
            telemetry = sample['frame_data']['telemetry']
            features.append([
                telemetry['speed'],
                telemetry['yaw'],
                telemetry['yaw_diff'],
                telemetry.get('accel', {}).get('x', 0.0),
                telemetry.get('angular_velocity', {}).get('y', 0.0)
            ])
        features = np.array(features)
        self.telemetry_mean = features.mean(axis=0)
        self.telemetry_std = features.std(axis=0)
        self.telemetry_std[self.telemetry_std == 0] = 1  # Avoid division by zero

    def _get_augmentation(self):
        """Get random augmentation transforms"""
        return T.Compose([
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.RandomAdjustSharpness(sharpness_factor=1.5, p=0.3),
            T.RandomHorizontalFlip(p=0.5),  # Flip image and steering angle
        ])

    def __len__(self):
        return len(self.samples)
    
    def _load_with_cache(self, path):
        """Load frame data with LRU cache"""
        if path in self.cache:
            return self.cache[path]
            
        data = np.load(path)
        
        # Implement simple LRU cache
        if len(self.cache) >= self.cache_size:
            self.cache.pop(next(iter(self.cache)))
        self.cache[path] = data
        return data

    def normalize_telemetry(self, values, feature_names):
        """Normalize telemetry values to [-1, 1] range"""
        normalized = []
        for value, name in zip(values, feature_names):
            min_val, max_val = self.telemetry_ranges[name]
            normalized.append(2.0 * (value - min_val) / (max_val - min_val) - 1.0)
        return np.array(normalized, dtype=np.float32)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        clip_dir = sample['clip_dir']
        frame_data = sample['frame_data']
        
        # Load stacked frame data
        frame_path = os.path.join(clip_dir, "processed", frame_data['processed_file'])
        stacked_frames = self._load_with_cache(frame_path)
        
        # Extract and normalize telemetry features
        telemetry = frame_data['telemetry']
        feature_names = ['speed', 'yaw', 'yaw_diff', 'accel_x', 'angular_velocity_y']
        telemetry_values = [
            telemetry[name] for name in feature_names
        ]
        telemetry_features = self.normalize_telemetry(telemetry_values, feature_names)
        
        # Extract target outputs (continuous values)
        targets = np.array([
            telemetry['steering'],  # -1.0 to 1.0
            telemetry['throttle'],  # 0.0 to 1.0
            telemetry['brake']      # 0.0 to 1.0
        ], dtype=np.float32)
        
        return {
            'frames': torch.from_numpy(stacked_frames).float().permute(2, 0, 1),
            'telemetry': torch.from_numpy(telemetry_features).float(),
            'targets': torch.from_numpy(targets).float(),
            'metadata': {
                'target_ranges': self.target_ranges,
                'telemetry_ranges': self.telemetry_ranges
            }
        }

    def _filter_samples(self, samples):
        """Filter out invalid or extreme samples"""
        filtered = []
        for sample in samples:
            telemetry = sample['frame_data']['telemetry']
            
            # Skip samples with extreme values
            if abs(telemetry['steering']) > 0.95:  # Near max steering
                continue
            if telemetry['speed'] < 2.0:  # Very slow/stationary
                continue
            
            filtered.append(sample)
        
        return filtered

def create_dataloaders(processed_dir, batch_size=32, train_split=0.8):
    """Create train and validation dataloaders"""
    dataset = DrivingDataset(processed_dir, transform=True)
    
    # Use fixed seed for reproducible splits
    generator = torch.Generator().manual_seed(42)
    
    # Stratified split based on steering angles
    steering_angles = [s['frame_data']['telemetry']['steering'] for s in dataset.samples]
    bins = pd.qcut(steering_angles, q=10, labels=False)  # 10 equal-sized bins
    
    train_idx, val_idx = train_test_split(
        np.arange(len(dataset)),
        test_size=1-train_split,
        stratify=bins,
        random_state=42
    )
    
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    
    # Use persistent workers and prefetch factor
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        generator=generator
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    return train_loader, val_loader 