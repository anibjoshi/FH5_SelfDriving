import socket
import struct

def debug_telemetry():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('', 5300))
    
    print("Listening for Forza telemetry on port 5300...")
    print("Press Ctrl+C to stop")
    
    last_yaw = None
    frame_count = 0
    
    try:
        while True:
            data, _ = sock.recvfrom(1024)
            
            try:
                # Controls
                steer = struct.unpack('b', data[320:321])[0]
                steer_normalized = steer/127.0
                throttle = struct.unpack('b', data[315:316])[0] * -1
                brake = struct.unpack('b', data[316:317])[0] * -1
                
                # Car motion data
                accel_x = struct.unpack('f', data[20:24])[0]
                accel_y = struct.unpack('f', data[24:28])[0]
                accel_z = struct.unpack('f', data[28:32])[0]
                
                vel_x = struct.unpack('f', data[32:36])[0]
                vel_y = struct.unpack('f', data[36:40])[0]
                vel_z = struct.unpack('f', data[40:44])[0]
                
                ang_vel_x = struct.unpack('f', data[44:48])[0]
                ang_vel_y = struct.unpack('f', data[48:52])[0]
                ang_vel_z = struct.unpack('f', data[52:56])[0]
                
                # Basic data
                speed = struct.unpack('f', data[256:260])[0]
                yaw = struct.unpack('f', data[56:60])[0]
                
                if last_yaw is not None:
                    yaw_diff = yaw - last_yaw
                    if yaw_diff > 180:
                        yaw_diff -= 360
                    elif yaw_diff < -180:
                        yaw_diff += 360
                    
                    print(f"Frame: {frame_count:5d}")
                    print(f"Controls: [T:{1 if throttle > 0 else 0} B:{1 if brake > 0 else 0} S:{steer_normalized:6.3f}]")
                    print(f"Speed: {speed:6.1f} m/s | Yaw: {yaw:8.3f}° | Diff: {yaw_diff:+8.3f}°")
                    print(f"Acceleration: [{accel_x:8.3f}, {accel_y:8.3f}, {accel_z:8.3f}]")
                    print(f"Velocity: [{vel_x:8.3f}, {vel_y:8.3f}, {vel_z:8.3f}]")
                    print(f"Angular Velocity: [{ang_vel_x:8.3f}, {ang_vel_y:8.3f}, {ang_vel_z:8.3f}]")
                    print("-" * 80)
                
                last_yaw = yaw
                frame_count += 1
                
            except struct.error as e:
                print(f"Error unpacking data: {e}")
                
    except KeyboardInterrupt:
        print("\nStopping telemetry debug...")
    finally:
        sock.close()

if __name__ == "__main__":
    debug_telemetry()