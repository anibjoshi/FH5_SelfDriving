import socket
import struct

class TelemetryReader:
    def __init__(self, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('', port))
        self.sock.setblocking(False)
        self.last_telemetry = None
    
    def get_telemetry(self):
        """Get telemetry data from UDP socket with continuous values"""
        try:
            data, _ = self.sock.recvfrom(1024)
            
            # Get raw control inputs
            steer = struct.unpack('b', data[320:321])[0]
            steer_normalized = round(steer/127.0, 3)  # -1.0 to 1.0
            
            # Get continuous throttle/brake values
            throttle_raw = -struct.unpack('b', data[315:316])[0]
            brake_raw = -struct.unpack('b', data[316:317])[0]
            throttle = max(0.0, min(1.0, throttle_raw / 127.0))
            brake = max(0.0, min(1.0, brake_raw / 127.0))
            
            # Car motion data
            speed = round(struct.unpack('f', data[256:260])[0], 3)
            yaw = round(struct.unpack('f', data[56:60])[0], 3)
            
            # Position and orientation
            pos_x = round(struct.unpack('f', data[8:12])[0], 3)
            pos_y = round(struct.unpack('f', data[12:16])[0], 3)
            pos_z = round(struct.unpack('f', data[16:20])[0], 3)
            
            # Acceleration
            accel_x = round(struct.unpack('f', data[20:24])[0], 3)
            accel_y = round(struct.unpack('f', data[24:28])[0], 3)
            accel_z = round(struct.unpack('f', data[28:32])[0], 3)
            
            # Velocity
            vel_x = round(struct.unpack('f', data[32:36])[0], 3)
            vel_y = round(struct.unpack('f', data[36:40])[0], 3)
            vel_z = round(struct.unpack('f', data[40:44])[0], 3)
            
            # Angular velocity
            ang_vel_x = round(struct.unpack('f', data[44:48])[0], 3)
            ang_vel_y = round(struct.unpack('f', data[48:52])[0], 3)
            ang_vel_z = round(struct.unpack('f', data[52:56])[0], 3)
            
            telemetry = {
                # Control inputs
                'throttle': throttle_raw,        # 0.0 to 1.0
                'brake': brake_raw,              # 0.0 to 1.0
                'steering': steer_normalized, # -1.0 to 1.0
                
                # Basic motion
                'speed': speed,
                'yaw': yaw,
                
                # Position
                'position': {
                    'x': pos_x,
                    'y': pos_y,
                    'z': pos_z
                },
                
                # Acceleration
                'acceleration': {
                    'x': accel_x,
                    'y': accel_y,
                    'z': accel_z
                },
                
                # Velocity
                'velocity': {
                    'x': vel_x,
                    'y': vel_y,
                    'z': vel_z
                },
                
                # Angular velocity
                'angular_velocity': {
                    'x': ang_vel_x,
                    'y': ang_vel_y,
                    'z': ang_vel_z
                }
            }
            
            self.last_telemetry = telemetry
            return telemetry
            
        except BlockingIOError:
            return self.last_telemetry or {
                'throttle': 0.0,
                'brake': 0.0,
                'steering': 0.0,
                'speed': 0.0,
                'yaw': 0.0,
                'position': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                'acceleration': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                'velocity': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                'angular_velocity': {'x': 0.0, 'y': 0.0, 'z': 0.0}
            }
    
    def close(self):
        self.sock.close() 