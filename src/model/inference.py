import torch
import numpy as np
import cv2
import mss
import vgamepad as vg
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

def inference(model_path: str):
    """Run inference loop"""
    agent = DrivingAgent(model_path)
    
    try:
        print("Starting inference... Press Ctrl+C to stop")
        while True:
            # Get and apply controls
            steering, throttle, brake = agent.step()
            
            # Print current controls
            print(f"\rSteering: {steering:+.2f} | Throttle: {throttle:.2f} | Brake: {brake:.2f}", end='')
            
    except KeyboardInterrupt:
        print("\nStopping inference...")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='Path to trained model')
    args = parser.parse_args()
    
    inference(args.model)