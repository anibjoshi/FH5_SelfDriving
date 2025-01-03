import numpy as np
import time
from src.utils.telemetry import TelemetryReader
import vgamepad as vg
import keyboard

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0
        
    def compute(self, error, dt):
        """Compute PID control value"""
        # Prevent division by zero
        if dt < 0.001:  # Less than 1ms
            dt = 0.001
            
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        
        output = (self.kp * error + 
                 self.ki * self.integral + 
                 self.kd * derivative)
        
        self.prev_error = error
        return output

class DummyDriver:
    def __init__(self):
        self.telemetry = TelemetryReader(5300)
        
        # PID controllers
        self.steering_pid = PIDController(kp=1.0, ki=0.1, kd=0.05)
        self.speed_pid = PIDController(kp=0.5, ki=0.1, kd=0.05)
        
        # Speed settings
        self.target_speed = 30.0  # m/s
        self.max_speed = 30.0     # Speed limit
        
        # Initialize virtual Xbox controller
        self.gamepad = vg.VX360Gamepad()
        
    def apply_controls(self, throttle, brake, steering):
        """Apply control inputs via virtual Xbox controller"""
        # Apply steering to left stick (-1.0 to 1.0)
        # Keep Y at 0 since we only need left/right steering
        self.gamepad.left_joystick_float(x_value_float=steering, y_value_float=0.0)
        
        # Apply throttle to RT and brake to LT (0.0 to 1.0)
        self.gamepad.right_trigger_float(value_float=throttle)  # RT for throttle
        self.gamepad.left_trigger_float(value_float=brake)      # LT for brake
        
        # Update the controller
        self.gamepad.update()
    
    def run(self):
        """Main control loop"""
        print("Starting dummy driver in 10 seconds...")
        print("Please switch to the game window...")
        print(f"Target speed: {self.target_speed:.1f} m/s ({self.target_speed * 3.6:.1f} km/h)")
        time.sleep(10)
        print("Running! Press 'Q' to quit")
        
        last_time = time.time()
        
        try:
            while True:
                current_time = time.time()
                dt = current_time - last_time
                last_time = current_time
                
                # Get telemetry
                telemetry = self.telemetry.get_telemetry()
                if not telemetry:
                    continue
                
                current_speed = telemetry['speed']
                
                # Speed control logic
                if current_speed >= self.max_speed:
                    # Above speed limit - coast or brake
                    throttle = 0
                    brake = 1.0 if current_speed > self.max_speed + 5 else 0.3
                else:
                    # Normal speed control
                    speed_error = self.target_speed - current_speed
                    speed_correction = self.speed_pid.compute(speed_error, dt)
                    
                    # Convert PID output to throttle/brake
                    if speed_correction > 0:
                        throttle = min(1.0, speed_correction)
                        brake = 0
                    else:
                        throttle = 0
                        brake = min(1.0, -speed_correction)
                
                # Random steering with smoothing
                steering = np.clip(np.random.normal(0, 0.3), -1, 1)
                
                # Apply controls
                self.apply_controls(throttle, brake, steering)
                
                # Print debug info
                print(f"\rSpeed: {current_speed:.1f} m/s ({current_speed * 3.6:.1f} km/h) | "
                      f"Throttle: {throttle:.2f} | Brake: {brake:.2f} | "
                      f"Steering: {steering:.2f}", end='')
                
                if keyboard.is_pressed('q'):
                    break
                
                time.sleep(0.01)
                
        finally:
            # Reset controller
            self.gamepad.reset()
            self.telemetry.close()

if __name__ == "__main__":
    driver = DummyDriver()
    driver.run()