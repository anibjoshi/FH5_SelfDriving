import cv2
import numpy as np
from mss import mss
import time
import tkinter as tk

def show_capture_area():
    # Create a reference window using tkinter
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.withdraw()  # Hide the tkinter window
    
    sct = mss()
    bounding_box = {
        'top': 100,
        'left': 0,
        'width': 1366,  # Standard 16:9 resolution
        'height': 768
    }
    
    # Create a reference window on the right side of the screen
    cv2.namedWindow('Reference Window')
    cv2.moveWindow('Reference Window', screen_width - 400, 0)
    
    reference = np.zeros((300, 400, 3), dtype=np.uint8)
    # Draw the capture area representation
    scale = min(380/1366, 280/768)  # Adjusted scale for new dimensions
    ref_width = int(1366 * scale)
    ref_height = int(768 * scale)
    start_x = (400 - ref_width) // 2
    start_y = (300 - ref_height) // 2
    
    # Draw scaled rectangle in reference window
    cv2.rectangle(reference, 
                 (start_x, start_y), 
                 (start_x + ref_width, start_y + ref_height),
                 (0, 255, 0), 2)
    
    print("\nCapture Area Alignment Tool")
    print("---------------------------")
    print("1. Position your game window")
    print("2. Align with the green rectangle dimensions:")
    print(f"   Width: {bounding_box['width']}")
    print(f"   Height: {bounding_box['height']}")
    print(f"   Top: {bounding_box['top']}")
    print(f"   Left: {bounding_box['left']}")
    print("3. Press 'q' to quit\n")
    
    try:
        while True:
            # Capture screen
            sct_img = sct.grab(bounding_box)
            frame = np.array(sct_img)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
            # Show the reference window
            cv2.putText(reference, "Capture Area", (start_x, start_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(reference, f"{bounding_box['width']}x{bounding_box['height']}", 
                       (start_x, start_y + ref_height + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow('Reference Window', reference)
            
            # Show the actual capture
            cv2.imshow('Capture', frame)
            
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
                
    finally:
        cv2.destroyAllWindows()
        root.destroy()

if __name__ == "__main__":
    show_capture_area() 