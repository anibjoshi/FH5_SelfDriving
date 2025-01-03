import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import wandb
from tqdm import tqdm

from .model import CNNModel
from src.data.dataset import DrivingDataset

def train(data_dir: str, save_dir: str, use_wandb: bool = True):
    """Train the PilotNet model"""
    
    # Initialize model
    model = CNNModel()
    model.to('cuda')
    
    # Create dataset and splits
    dataset = DrivingDataset(data_dir, model_type='cnn')
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Initialize wandb
    if use_wandb:
        wandb.init(project="self-driving", name="pilotnet")
        wandb.watch(model)
    
    # Training loop
    num_epochs = 100
    best_val_loss = float('inf')
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_losses = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
            frames = batch['frames'].cuda()
            telemetry = batch['telemetry'].cuda()
            targets = batch['targets'].cuda()
            
            # Forward pass
            predictions = model(frames, telemetry)
            
            # Calculate loss
            steering_loss = nn.MSELoss()(predictions[:, 0], targets[:, 0])
            throttle_loss = nn.MSELoss()(predictions[:, 1], targets[:, 1])
            brake_loss = nn.MSELoss()(predictions[:, 2], targets[:, 2])
            
            loss = steering_loss + 0.5 * (throttle_loss + brake_loss)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_losses.append({
                'total': loss.item(),
                'steering': steering_loss.item(),
                'throttle': throttle_loss.item(),
                'brake': brake_loss.item()
            })
        
        # Validation
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
                frames = batch['frames'].cuda()
                telemetry = batch['telemetry'].cuda()
                targets = batch['targets'].cuda()
                
                predictions = model(frames, telemetry)
                
                steering_loss = nn.MSELoss()(predictions[:, 0], targets[:, 0])
                throttle_loss = nn.MSELoss()(predictions[:, 1], targets[:, 1])
                brake_loss = nn.MSELoss()(predictions[:, 2], targets[:, 2])
                
                loss = steering_loss + 0.5 * (throttle_loss + brake_loss)
                
                val_losses.append({
                    'total': loss.item(),
                    'steering': steering_loss.item(),
                    'throttle': throttle_loss.item(),
                    'brake': brake_loss.item()
                })
        
        # Calculate averages
        train_avg = {k: sum(d[k] for d in train_losses) / len(train_losses) 
                    for k in train_losses[0].keys()}
        val_avg = {k: sum(d[k] for d in val_losses) / len(val_losses) 
                  for k in val_losses[0].keys()}
        
        # Log metrics
        if use_wandb:
            wandb.log({
                **{f"train_{k}": v for k, v in train_avg.items()},
                **{f"val_{k}": v for k, v in val_avg.items()},
                "epoch": epoch
            })
        
        # Save best model
        if val_avg['total'] < best_val_loss:
            best_val_loss = val_avg['total']
            torch.save(model.state_dict(), save_dir / 'best_model.pt')
        
        # Print metrics
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("Train:", {k: f"{v:.4f}" for k, v in train_avg.items()})
        print("Val:", {k: f"{v:.4f}" for k, v in val_avg.items()})

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', help='Path to data directory')
    parser.add_argument('save_dir', help='Path to save models')
    parser.add_argument('--no-wandb', action='store_true', help='Disable wandb logging')
    args = parser.parse_args()
    
    train(args.data_dir, args.save_dir, use_wandb=not args.no_wandb) 