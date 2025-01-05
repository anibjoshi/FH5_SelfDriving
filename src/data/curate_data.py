import cv2
import numpy as np
import json
import os
from tqdm import tqdm
import tkinter as tk
from tkinter import ttk
import shutil
from pathlib import Path
from src.config import *

class DataCurator:
    def __init__(self, session_dir):
        # Convert session_dir to full path relative to RAW_DATA_DIR
        self.session_dir = os.path.join(RAW_DATA_DIR, session_dir)
        self.session_dir = os.path.abspath(self.session_dir)
        
        if not os.path.exists(self.session_dir):
            raise FileNotFoundError(f"Session directory not found: {self.session_dir}")
        
        # Load data from JSONL
        self.frame_data = []
        self.session_info = None
        self.load_data()
        
        self.current_idx = 0
        
        # Playback settings
        self.playing = False
        self.play_speed = 1
        
        # Clip settings
        self.fps = 30  # Estimated FPS
        self.target_clip_duration = 120  # 2 minutes in seconds
        self.target_clip_frames = self.target_clip_duration * self.fps
        self.clip_start = None
        self.clips = []  # List of (start_frame, end_frame) tuples
        
        # Setup UI
        self.setup_ui()
    
    def load_data(self):
        """Load data from JSONL file"""
        jsonl_path = os.path.join(self.session_dir, "data.jsonl")
        print(f"Loading data from: {jsonl_path}")
        self.frames_dir = os.path.join(self.session_dir, "frames")
        
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"Data file not found: {jsonl_path}")
        if not os.path.exists(self.frames_dir):
            raise FileNotFoundError(f"Frames directory not found: {self.frames_dir}")
        
        print("Loading session data...")
        with open(jsonl_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                if data['type'] == 'session_info':
                    self.session_info = data
                elif data['type'] == 'frame':
                    self.frame_data.append(data)
        
        print(f"Loaded {len(self.frame_data)} frames")
        
    def setup_ui(self):
        self.root = tk.Tk()
        self.root.title("Data Curator")
        
        # Main controls
        controls = ttk.Frame(self.root)
        controls.pack(fill='x', padx=5, pady=5)
        
        # Navigation controls
        nav_frame = ttk.LabelFrame(controls, text="Navigation")
        nav_frame.pack(side='left', padx=5)
        
        ttk.Button(nav_frame, text="◀◀ -300", 
                  command=lambda: self.jump_frames(-300)).pack(side='left', padx=2)
        ttk.Button(nav_frame, text="◀◀", 
                  command=self.prev_frame).pack(side='left', padx=2)
        ttk.Button(nav_frame, text="▶/∥", 
                  command=self.toggle_play).pack(side='left', padx=2)
        ttk.Button(nav_frame, text="▶▶", 
                  command=self.next_frame).pack(side='left', padx=2)
        ttk.Button(nav_frame, text="300 ▶▶", 
                  command=lambda: self.jump_frames(300)).pack(side='left', padx=2)
        
        # Playback speed
        speed_frame = ttk.LabelFrame(controls, text="Speed")
        speed_frame.pack(side='left', padx=5)
        
        self.speed_var = tk.StringVar(value="1x")
        for speed in ["0.5x", "1x", "2x", "4x", "8x"]:
            ttk.Radiobutton(speed_frame, text=speed, value=speed,
                          variable=self.speed_var,
                          command=self.change_speed).pack(side='left', padx=2)
        
        # Clip controls
        clip_frame = ttk.LabelFrame(controls, text="Clip Controls")
        clip_frame.pack(side='left', padx=5)
        
        ttk.Button(clip_frame, text="Start Clip (s)", 
                  command=self.mark_clip_start).pack(side='left', padx=2)
        ttk.Button(clip_frame, text="End Clip (e)", 
                  command=self.mark_clip_end).pack(side='left', padx=2)
        
        # Save button
        ttk.Button(controls, text="Save Clips", 
                  command=self.save_clips).pack(side='right', padx=5)
        
        # Progress bar and info
        info_frame = ttk.Frame(self.root)
        info_frame.pack(fill='x', padx=5, pady=5)
        
        self.progress_var = tk.DoubleVar(value=0)
        self.progress = ttk.Progressbar(info_frame, variable=self.progress_var,
                                      maximum=len(self.frame_data)-1)
        self.progress.pack(fill='x')
        
        # Info labels
        self.info_label = ttk.Label(info_frame, text="")
        self.info_label.pack()
        
        self.clip_info = ttk.Label(info_frame, text="")
        self.clip_info.pack()
        
        # Keyboard bindings
        self.root.bind('s', lambda e: self.mark_clip_start())
        self.root.bind('e', lambda e: self.mark_clip_end())
        self.root.bind('<Left>', lambda e: self.prev_frame())
        self.root.bind('<Right>', lambda e: self.next_frame())
        self.root.bind('<space>', lambda e: self.toggle_play())
        self.root.bind('<Prior>', lambda e: self.jump_frames(-300))
        self.root.bind('<Next>', lambda e: self.jump_frames(300))
        
        # Start display
        self.update_display()
    
    def jump_frames(self, offset):
        self.playing = False
        self.current_idx = max(0, min(self.current_idx + offset, 
                                    len(self.frame_data) - 1))
        self.update_display()
    
    def mark_clip_start(self):
        self.clip_start = self.current_idx
        self.update_display()
    
    def mark_clip_end(self):
        if self.clip_start is not None and self.current_idx > self.clip_start:
            duration = (self.current_idx - self.clip_start) / self.fps
            if duration > self.target_clip_duration:
                print(f"Warning: Clip duration ({duration:.1f}s) exceeds target ({self.target_clip_duration}s)")
            
            self.clips.append((self.clip_start, self.current_idx))
            print(f"Clip marked: {len(self.clips)} "
                  f"({duration:.1f}s, {self.current_idx - self.clip_start} frames)")
            self.clip_start = None
        self.update_display()
    
    def update_display(self):
        if self.current_idx >= len(self.frame_data):
            self.playing = False
            return
        
        # Load and display frame
        frame_data = self.frame_data[self.current_idx]
        frame_path = os.path.join(self.frames_dir, frame_data['file'])
        frame = cv2.imread(frame_path)
        
        # Show telemetry data
        telemetry = frame_data['telemetry']
        cv2.putText(frame, f"Frame: {self.current_idx}/{len(self.frame_data)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Steering: {telemetry['steering']:.2f}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Speed: {telemetry['speed']*3.6:.0f}km/h", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show clip information
        if self.clip_start is not None:
            duration = (self.current_idx - self.clip_start) / self.fps
            cv2.putText(frame, f"Recording Clip: {duration:.1f}s", 
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Show if frame is in an existing clip
        for i, (start, end) in enumerate(self.clips):
            if start <= self.current_idx <= end:
                cv2.putText(frame, f"In Clip {i+1}", 
                           (frame.shape[1]-200, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Frame Review', frame)
        cv2.waitKey(1)
        
        # Update progress and info
        self.progress_var.set(self.current_idx)
        self.info_label.config(
            text=f"Frame {self.current_idx}/{len(self.frame_data)} | "
                 f"Time: {self.current_idx/self.fps:.1f}s")
        self.clip_info.config(
            text=f"Clips marked: {len(self.clips)} | "
                 f"Current clip duration: {(self.current_idx - (self.clip_start or self.current_idx))/self.fps:.1f}s")
        
        if self.playing:
            self.current_idx += self.play_speed
            self.root.after(1, self.update_display)
    
    def toggle_play(self):
        self.playing = not self.playing
        if self.playing:
            self.update_display()
    
    def next_frame(self):
        self.playing = False
        self.current_idx = min(self.current_idx + 1, len(self.frame_data) - 1)
        self.update_display()
    
    def prev_frame(self):
        self.playing = False
        self.current_idx = max(self.current_idx - 1, 0)
        self.update_display()
    
    def change_speed(self):
        speed = float(self.speed_var.get().replace('x', ''))
        self.play_speed = int(speed * 2)  # Adjust multiplier as needed
    
    def save_clips(self):
        if not self.clips:
            print("No clips marked!")
            return
        
        # Create session-specific directory in curated_data
        session_name = os.path.basename(self.session_dir)
        curated_dir = os.path.join(CURATED_DATA_DIR, session_name)
        os.makedirs(curated_dir, exist_ok=True)
        
        print("\nSaving clips...")
        for clip_idx, (start_frame, end_frame) in enumerate(tqdm(self.clips)):
            clip_dir = os.path.join(curated_dir, f"clip_{clip_idx:03d}")
            frames_dir = os.path.join(clip_dir, "frames")
            os.makedirs(frames_dir, exist_ok=True)
            
            # Get clip frames
            clip_frames = self.frame_data[start_frame:end_frame+1]
            
            # Open clip data file
            with open(os.path.join(clip_dir, "data.jsonl"), 'w') as f:
                # Write session info
                clip_session_info = self.session_info.copy()
                clip_session_info['clip_number'] = clip_idx
                f.write(json.dumps(clip_session_info) + '\n')
                
                # Copy frames and write data
                for new_idx, frame_data in enumerate(clip_frames):
                    # Copy frame file
                    old_path = os.path.join(self.frames_dir, frame_data['file'])
                    new_filename = f"frame_{new_idx:06d}.jpg"
                    new_path = os.path.join(frames_dir, new_filename)
                    shutil.copy2(old_path, new_path)
                    
                    # Write frame data
                    frame_entry = frame_data.copy()
                    frame_entry['file'] = new_filename
                    frame_entry['frame'] = new_idx
                    f.write(json.dumps(frame_entry) + '\n')
                
                # Write clip end info
                clip_end = {
                    'type': 'clip_end',
                    'total_frames': len(clip_frames),
                    'duration': clip_frames[-1]['timestamp'] - clip_frames[0]['timestamp']
                }
                f.write(json.dumps(clip_end) + '\n')
        
        print(f"\nSaved {len(self.clips)} clips:")
        for i, (start, end) in enumerate(self.clips):
            duration = (end - start) / self.fps
            print(f"Clip {i}: {duration:.1f}s ({end-start} frames)")
        print(f"Saved to: {curated_dir}")
        
        self.root.quit()
    
    def run(self):
        self.root.mainloop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    session_dir = "collected_data/20250102_145319"  # Replace with your session directory
    curator = DataCurator(session_dir)
    curator.run() 