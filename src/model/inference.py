import torch
import numpy as np
import cv2
import mss
import vgamepad as vg
import random
import time
from pathlib import Path
from src.model.cnn.model import CNNModel
from src.utils.telemetry import TelemetryReader
from src.config import *

class ScreenCapture:
    def __init__(self):
        self.sct = mss.mss()
        # Get primary monitor
        self.monitor = self.sct.monitors[1]
        
        # Calculate center crop dimensions (16:9 aspect ratio)
        self.width = self.monitor['width']
        self.height = self.monitor['height']
        target_width = int(self.height * 16 / 9)
        
        # Center the capture area
        self.left = (self.width - target_width) // 2
        self.capture_area = {
            'left': self.left,
            'top': 0,
            'width': target_width,
            'height': self.height
        }

    def capture(self):
        """Capture and preprocess a frame"""
        # Capture screen
        frame = np.array(self.sct.grab(self.capture_area))
        
        # Convert from BGRA to BGR
        frame = frame[:, :, :3]
        
        # Convert to YUV color space (as used in training)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        
        # Crop top portion (same as training)
        height = frame.shape[0]
        crop_pixels = int(height * CROP_TOP_PERCENT / 100)
        frame = frame[crop_pixels:, :]
        
        # Resize to PilotNet dimensions
        frame = cv2.resize(frame, (200, 66))
        
        return frame

class VirtualController:
    def __init__(self):
        self.gamepad = vg.VX360Gamepad()
    
    def apply_controls(self, steering, throttle, brake):
        """Apply controls to virtual xbox controller"""
        # Steering: map [-1, 1] to [-32768, 32767]
        steering_val = int(steering * 32767)
        self.gamepad.left_joystick_float(x_value_float=steering, y_value_float=0.0)
        
        # Throttle: map [0, 1] to [0, 255]
        self.gamepad.right_trigger_float(throttle)
        
        # Brake: map [0, 1] to [0, 255]
        self.gamepad.left_trigger_float(brake)
        
        # Update controller
        self.gamepad.update()

class DrivingAgent:
    def __init__(self, model_path: str):
        # Initialize model
        self.model = CNNModel()
        
        # Load trained weights
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        self.model.cuda()
        
        # Initialize capture, telemetry and controller
        self.screen_capture = ScreenCapture()
        self.telemetry = TelemetryReader()
        self.controller = VirtualController()
        
        # Frame buffer for stacking
        self.frame_buffer = []
        
        # YUV normalization values (from training)
        self.yuv_mean = torch.tensor([0.5, 0, 0]).view(3, 1, 1)
        self.yuv_std = torch.tensor([0.5, 1, 1]).view(3, 1, 1)
    
    def process_frame(self):
        """Capture, process frame and return control outputs"""
        # Capture and preprocess frame
        frame = self.screen_capture.capture()
        
        # Convert to tensor and normalize
        frame = torch.from_numpy(frame).float()
        frame = frame.permute(2, 0, 1)  # HWC to CHW
        frame = frame / 255.0  # Scale to [0, 1]
        
        # Normalize with YUV mean/std
        frame = (frame - self.yuv_mean.to(frame.device)) / self.yuv_std.to(frame.device)
        
        # Add to frame buffer
        self.frame_buffer.append(frame)
        if len(self.frame_buffer) > FRAME_STACK_SIZE:
            self.frame_buffer.pop(0)
        
        # If we don't have enough frames yet, return neutral controls
        if len(self.frame_buffer) < FRAME_STACK_SIZE:
            return 0.0, 0.0, 0.0
        
        # Stack frames
        stacked_frames = torch.stack(self.frame_buffer, dim=0)
        
        # Get telemetry
        telemetry = self.telemetry.get_telemetry()
        telemetry = torch.tensor([
            telemetry['speed'],
            telemetry['yaw'],
            telemetry['yaw_diff'],
            telemetry.get('accel_x', 0.0),
            telemetry.get('angular_velocity_y', 0.0)
        ], dtype=torch.float32)
        
        # Add batch dimension and move to GPU
        frames = stacked_frames.unsqueeze(0).cuda()
        telemetry = telemetry.unsqueeze(0).cuda()
        
        # Get model predictions
        with torch.no_grad():
            predictions = self.model(frames, telemetry)
        
        # Extract controls
        steering = predictions[0, 0].item()  # [-1, 1]
        throttle = predictions[0, 1].item()  # [0, 1]
        brake = predictions[0, 2].item()     # [0, 1]
        
        return steering, throttle, brake
    
    def step(self):
        """Capture frame, get predictions, and apply controls"""
        steering, throttle, brake = self.process_frame()
        self.controller.apply_controls(steering, throttle, brake)
        return steering, throttle, brake

class RandomAgent:
    """Generate smooth random controls for testing"""
    def __init__(self):
        self.controller = VirtualController()
        self.current_steering = 0.0
        self.current_throttle = 0.0
        self.current_brake = 0.0
        
        # Control smoothing
        self.steering_step = 0.1
        self.pedal_step = 0.1
        
        # Target values
        self.target_steering = 0.0
        self.target_throttle = 0.0
        self.target_brake = 0.0
        
        # Update targets periodically
        self.last_target_update = time.time()
        self.target_update_interval = 2.0  # seconds
    
    def _update_targets(self):
        """Randomly update target values"""
        self.target_steering = random.uniform(-1.0, 1.0)
        
        # Randomly choose between acceleration and braking
        if random.random() < 0.8:  # 80% chance of throttle
            self.target_throttle = random.uniform(0.3, 1.0)
            self.target_brake = 0.0
        else:
            self.target_throttle = 0.0
            self.target_brake = random.uniform(0.3, 1.0)
    
    def _smooth_control(self, current, target, step):
        """Smoothly adjust current value toward target"""
        if current < target:
            return min(current + step, target)
        else:
            return max(current - step, target)
    
    def step(self):
        """Update and apply controls"""
        # Update targets periodically
        current_time = time.time()
        if current_time - self.last_target_update > self.target_update_interval:
            self._update_targets()
            self.last_target_update = current_time
        
        # Smooth controls
        self.current_steering = self._smooth_control(
            self.current_steering, self.target_steering, self.steering_step)
        self.current_throttle = self._smooth_control(
            self.current_throttle, self.target_throttle, self.pedal_step)
        self.current_brake = self._smooth_control(
            self.current_brake, self.target_brake, self.pedal_step)
        
        # Apply controls
        self.controller.apply_controls(
            self.current_steering, 
            self.current_throttle, 
            self.current_brake
        )
        
        return self.current_steering, self.current_throttle, self.current_brake

def inference(model_path: str = None):
    """Run inference loop with either trained model or random agent"""
    if model_path:
        agent = DrivingAgent(model_path)
        print("Using trained model for control")
    else:
        agent = RandomAgent()
        print("Using random agent for control")
    
    print("\nWaiting 10 seconds to start...")
    time.sleep(10)
    print("Starting! Press Ctrl+C to stop")
    
    try:
        while True:
            # Get and apply controls
            steering, throttle, brake = agent.step()
            
            # Print current controls
            print(f"\rSteering: {steering:+.2f} | Throttle: {throttle:.2f} | Brake: {brake:.2f}", end='')
            
            # Small sleep to prevent maxing CPU
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\nStopping inference...")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Path to trained model (optional)')
    args = parser.parse_args()
    
    inference(args.model)