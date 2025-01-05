import cv2
import numpy as np
from mss import mss
import time
import os
from datetime import datetime
import json
import tkinter as tk
from src.utils.telemetry import TelemetryReader
from src.config import *

class DataCollector:
    def __init__(self, output_dir=RAW_DATA_DIR):
        # Screen capture setup
        self.sct = mss()
        self.bounding_box = {
            'top': CAPTURE_TOP,
            'left': CAPTURE_LEFT,
            'width': CAPTURE_WIDTH,
            'height': CAPTURE_HEIGHT
        }
        
        # Create reference window
        root = tk.Tk()
        self.screen_width = root.winfo_screenwidth()
        root.withdraw()
        
        # Setup reference window
        cv2.namedWindow('Reference')
        cv2.moveWindow('Reference', self.screen_width - 400, 0)
        
        self._setup_reference_window()
        
        # Setup telemetry
        self.telemetry_reader = TelemetryReader(UDP_PORT)
        
        # Setup output directory
        self.session_dir = os.path.join(output_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.frames_dir = os.path.join(self.session_dir, "frames")
        os.makedirs(self.frames_dir, exist_ok=True)
        
        # Initialize tracking
        self.start_time = time.time()
        self.frame_count = 0
        
        # Setup data file
        self.data_file = open(os.path.join(self.session_dir, "data.jsonl"), 'w')
        
        # Write session info
        self.session_info = {
            'type': 'session_info',
            'timestamp': datetime.now().isoformat(),
            'raw_shape': [CAPTURE_HEIGHT, CAPTURE_WIDTH, 3],
            'camera_config': {
                'view': 'hood',
                'capture_width': CAPTURE_WIDTH,
                'capture_height': CAPTURE_HEIGHT
            }
        }
        self.data_file.write(json.dumps(self.session_info) + '\n')

    def _setup_reference_window(self):
        """Setup the reference window showing capture area"""
        self.reference = np.zeros((300, 400, 3), dtype=np.uint8)
        
        # Calculate scaled dimensions for reference window
        scale = min(380/self.bounding_box['width'], 280/self.bounding_box['height'])
        ref_width = int(self.bounding_box['width'] * scale)
        ref_height = int(self.bounding_box['height'] * scale)
        start_x = (400 - ref_width) // 2
        start_y = (300 - ref_height) // 2
        
        # Draw scaled rectangle in reference window
        cv2.rectangle(self.reference, 
                     (start_x, start_y), 
                     (start_x + ref_width, start_y + ref_height),
                     (0, 255, 0), 2)
        
        cv2.putText(self.reference, "Capture Area", (start_x, start_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(self.reference, f"{self.bounding_box['width']}x{self.bounding_box['height']}", 
                   (start_x, start_y + ref_height + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    def capture_frame(self):
        """Capture a frame and its telemetry data"""
        # Capture screen area
        sct_img = self.sct.grab(self.bounding_box)
        frame = np.array(sct_img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        
        # Get telemetry
        telemetry = self.telemetry_reader.get_telemetry()
        
        
        # Save frame
        frame_filename = f"frame_{self.frame_count:06d}.jpg"
        cv2.imwrite(os.path.join(self.frames_dir, frame_filename), frame)
        
        # Write frame data
        frame_data = {
            'type': 'frame',
            'frame': self.frame_count,
            'timestamp': round(time.time() - self.start_time, 3),
            'file': frame_filename,
            'telemetry': telemetry
        }
        self.data_file.write(json.dumps(frame_data) + '\n')
        self.data_file.flush()  # Ensure data is written immediately
        
        self.frame_count += 1
        return frame, telemetry

    def run(self):
        """Main collection loop"""
        print("Starting in 10 seconds... Make sure:")
        print("1. Hood camera view is active")
        print("2. Game window aligns with reference box")
        print("3. Forza telemetry UDP output is enabled")
        time.sleep(10)
        print("Recording... Press 'q' to stop")
        
        try:
            while True:
                frame, telemetry = self.capture_frame()
                
                # Show minimal info overlay
                fps = self.frame_count / (time.time() - self.start_time)
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, "Recording...", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Show windows
                cv2.imshow('Recording', frame)
                cv2.imshow('Reference', self.reference)
                
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break
                
        finally:
            # Write session end info
            session_end = {
                'type': 'session_end',
                'total_frames': self.frame_count,
                'duration': round(time.time() - self.start_time, 3)
            }
            self.data_file.write(json.dumps(session_end) + '\n')
            
            # Cleanup
            self.data_file.close()
            cv2.destroyAllWindows()
            print(f"\nRecording finished!")
            print(f"Frames: {self.frame_count}")
            print(f"Avg FPS: {self.frame_count / (time.time() - self.start_time):.1f}")
            print(f"Saved to: {self.session_dir}")

if __name__ == "__main__":
    collector = DataCollector()
    collector.run() 