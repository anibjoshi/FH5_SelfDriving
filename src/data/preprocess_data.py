import cv2
import numpy as np
import json
import os
from tqdm import tqdm
from collections import deque
from pathlib import Path
from src.config import *

def compute_dataset_stats(clip_dir, sample_size=100):
    """Compute dataset mean and std for normalization"""
    with open(os.path.join(clip_dir, "data.jsonl"), 'r') as f:
        frames = [data for data in map(json.loads, f) if data['type'] == 'frame']
    
    # Randomly sample frames
    sample_indices = np.random.choice(len(frames), min(sample_size, len(frames)), replace=False)
    
    # Collect pixel values in YUV space
    pixels = []
    for idx in sample_indices:
        frame_path = os.path.join(clip_dir, "frames", frames[idx]['file'])
        frame = cv2.imread(frame_path)
        if frame is not None:
            # Convert to YUV color space
            frame_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            pixels.extend(frame_yuv.reshape(-1, 3))
    
    pixels = np.array(pixels)
    mean = pixels.mean(axis=0) / 255.0
    std = pixels.std(axis=0) / 255.0
    
    return mean, std

def preprocess_frame(frame, mean, std):
    """
    Preprocess a single frame following PilotNet paper:
    - Convert to YUV
    - Crop top portion
    - Resize to 66x200
    - Normalize
    """
    height = frame.shape[0]
    crop_pixels = int(height * CROP_TOP_PERCENT / 100)
    
    # Convert to YUV color space
    frame_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    
    # Crop top portion
    cropped = frame_yuv[crop_pixels:, :]
    
    # Resize to PilotNet dimensions
    resized = cv2.resize(cropped, (200, 66))
    
    # Normalize with mean and std
    normalized = (resized / 255.0 - mean) / std
    
    return cropped, normalized

def process_telemetry(frame_data, frame_buffer_data):
    """Extract relevant telemetry data for driving behavior"""
    telemetry = frame_data['telemetry']
    
    # Calculate yaw difference across the stack
    if frame_buffer_data:
        start_yaw = frame_buffer_data[0]['telemetry']['yaw']
        yaw_diff = telemetry['yaw'] - start_yaw
        if yaw_diff > 180: yaw_diff -= 360
        elif yaw_diff < -180: yaw_diff += 360
        yaw_rate = yaw_diff / len(frame_buffer_data)
    else:
        yaw_rate = 0.0
    
    return {
        'throttle': float(telemetry['throttle']),  # 0.0 to 1.0
        'brake': float(telemetry['brake']),        # 0.0 to 1.0
        'steering': float(telemetry['steering']),  # -1.0 to 1.0
        'speed': float(telemetry['speed']),
        'yaw': float(telemetry['yaw']),
        'yaw_diff': float(yaw_rate),
        'accel_x': float(telemetry.get('accel_x', 0.0)),
        'angular_velocity_y': float(telemetry.get('angular_velocity_y', 0.0))
    }

def process_clip(clip_dir: str):
    """Process all frames in a curated clip"""
    if not os.path.exists(clip_dir):
        raise FileNotFoundError(f"Clip directory not found: {clip_dir}")
    
    # Compute dataset statistics
    print("Computing dataset statistics...")
    mean, std = compute_dataset_stats(clip_dir)
    
    # Load clip data
    with open(os.path.join(clip_dir, "data.jsonl"), 'r') as f:
        frames = []
        clip_info = None
        for line in f:
            data = json.loads(line)
            if data['type'] == 'session_info':
                clip_info = data
            elif data['type'] == 'frame':
                frames.append(data)
    
    # Create processed directory structure
    session_name = os.path.basename(os.path.dirname(clip_dir))
    clip_name = os.path.basename(clip_dir)
    processed_base = os.path.join(PROCESSED_DATA_DIR, session_name, clip_name)
    
    cropped_dir = os.path.join(processed_base, "cropped")
    processed_dir = os.path.join(processed_base, "processed")
    os.makedirs(cropped_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    # Process all frames
    processed_frames = []
    print(f"Processing {len(frames)} frames from {clip_name}...")
    
    # Initialize frame buffers for both images and data
    frame_buffer = deque(maxlen=FRAME_STACK_SIZE)
    frame_data_buffer = deque(maxlen=FRAME_STACK_SIZE)
    
    for frame_data in tqdm(frames):
        # Load and process frame
        frame = cv2.imread(os.path.join(clip_dir, "frames", frame_data['file']))
        if frame is None:
            print(f"Warning: Could not load frame {frame_data['file']}")
            continue
            
        cropped, processed = preprocess_frame(frame, mean, std)
        
        # Save cropped frame as JPG
        cropped_filename = f"cropped_{frame_data['frame']:06d}.jpg"
        cv2.imwrite(os.path.join(cropped_dir, cropped_filename), cropped)
        
        # Add to frame buffers
        frame_buffer.append(processed)
        frame_data_buffer.append(frame_data)
        
        # Create stacked frames
        if len(frame_buffer) < FRAME_STACK_SIZE:
            padding = [frame_buffer[0]] * (FRAME_STACK_SIZE - len(frame_buffer))
            stack = padding + list(frame_buffer)
            data_padding = [frame_data_buffer[0]] * (FRAME_STACK_SIZE - len(frame_data_buffer))
            data_stack = data_padding + list(frame_data_buffer)
        else:
            stack = list(frame_buffer)
            data_stack = list(frame_data_buffer)
        
        stacked = np.stack(stack, axis=-1)
        
        # Save processed stacked frames
        processed_filename = f"processed_{frame_data['frame']:06d}.npy"
        np.save(os.path.join(processed_dir, processed_filename), stacked)
        
        # Extract relevant telemetry
        telemetry = process_telemetry(frame_data, data_stack[:-1])
        
        # Update frame data
        frame_data = frame_data.copy()
        frame_data['cropped_file'] = cropped_filename
        frame_data['processed_file'] = processed_filename
        frame_data['stack_size'] = FRAME_STACK_SIZE
        frame_data['actual_frames'] = len(frame_buffer)
        frame_data['padded'] = len(frame_buffer) < FRAME_STACK_SIZE
        frame_data['telemetry'] = telemetry
        processed_frames.append(frame_data)
    
    # Save processed clip metadata
    processed_info = {
        'clip_info': clip_info,
        'frames': processed_frames,
        'preprocessing': {
            'crop_top_percent': CROP_TOP_PERCENT,
            'target_size': [66, 200],  # PilotNet dimensions
            'color_space': 'YUV',
            'channels': 3 * FRAME_STACK_SIZE,
            'normalization': {
                'mean': mean.tolist(),
                'std': std.tolist()
            },
            'frame_stack_size': FRAME_STACK_SIZE
        }
    }
    
    with open(os.path.join(processed_base, "processed_data.json"), 'w') as f:
        json.dump(processed_info, f, indent=2)
    
    print(f"Processed clip saved to: {processed_base}")

def process_session(session_dir: str):
    """Process all clips in a curated session directory"""
    # Convert session_dir to full path relative to CURATED_DATA_DIR
    session_dir = os.path.join(CURATED_DATA_DIR, session_dir)
    session_dir = os.path.abspath(session_dir)
    
    if not os.path.exists(session_dir):
        raise FileNotFoundError(f"Session directory not found: {session_dir}")
    
    # Process each clip in the session
    clips = [d for d in os.listdir(session_dir) 
            if os.path.isdir(os.path.join(session_dir, d)) and d.startswith('clip_')]
    
    if not clips:
        raise ValueError(f"No clips found in session directory: {session_dir}")
    
    print(f"Found {len(clips)} clips in {session_dir}")
    for clip in sorted(clips):
        clip_dir = os.path.join(session_dir, clip)
        process_clip(clip_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('session_dir', help='Path to curated session directory')
    args = parser.parse_args()
    
    process_session(args.session_dir) 