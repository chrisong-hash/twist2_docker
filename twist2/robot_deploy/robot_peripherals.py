#!/usr/bin/env python3
"""
Robot Peripherals Controller - Video streaming and neck control for TWIST2.
Runs on the robot alongside sim2real.sh

This script handles:
1. ZED Mini video streaming to PICO VR headset
2. Neck motor control via Dynamixel (from Redis commands)

Usage:
  python3 robot_peripherals.py [options]

Options:
  --redis HOST      Redis server IP (default: 192.168.50.164)
  --yaw_center N    Yaw motor center position (default: 2048)
  --pitch_center N  Pitch motor center position (default: 2048)
  --no_video        Disable video streaming
  --no_neck         Disable neck control
  --device DEV      Serial device for Dynamixel (default: /dev/ttyUSB0)
"""

import struct
import time
import socket
import threading
import subprocess
import argparse
import math
import signal
import sys
import json
import os
import glob
import logging
from datetime import datetime

# ============== Logging Setup ==============
def setup_logging(log_dir="~/logs", max_logs=100):
    """Setup logging with rotation (keep max_logs newest files)"""
    log_dir = os.path.expanduser(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    
    # Cleanup old logs (keep newest max_logs)
    log_pattern = os.path.join(log_dir, "peripheral_*.log")
    existing_logs = sorted(glob.glob(log_pattern), key=os.path.getmtime)
    if len(existing_logs) > max_logs:
        for old_log in existing_logs[:-max_logs]:
            try:
                os.remove(old_log)
            except:
                pass
    
    # Note: actual logging goes to stdout which sim2real.sh redirects to file
    print(f"[Log] Log directory: {log_dir} ({len(existing_logs)} existing logs)")

# Redis watchdog settings
REDIS_WATCHDOG_TIMEOUT = 5.0  # seconds before shutdown on Redis disconnect

# ============== Configuration ==============
# Video settings
TARGET_WIDTH = 2560  # Side-by-side stereo
TARGET_HEIGHT = 720
COMMAND_PORT = 13579
STREAM_PORT = 12345
FRAMERATE = 30
BITRATE = 4000000

# Neck settings
DEVICENAME = '/dev/ttyUSB0'
BAUDRATE = 57600
ID_YAW = 0
ID_PITCH = 1
DEFAULT_YAW_CENTER = 1338
DEFAULT_PITCH_CENTER = 695

# Dynamixel addresses
ADDR_TORQUE_ENABLE = 64
ADDR_OPERATING_MODE = 11
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132
ADDR_PROFILE_VELOCITY = 112
ADDR_HARDWARE_ERROR = 70
PROTOCOL_VERSION = 2.0
# Position limits (from physical calibration)
YAW_MIN = 514
YAW_MAX = 2095
PITCH_MIN = 588
PITCH_MAX = 1626

# ============== Globals ==============
running = True
USE_ZED_SDK = False

try:
    import pyzed.sl as sl
    USE_ZED_SDK = True
except ImportError:
    pass

import cv2
import numpy as np


# ============== Video Server ==============
# Camera auto-restart settings
CAMERA_FAIL_THRESHOLD = 30  # Restart after this many consecutive grab failures
CAMERA_RESTART_DELAY = 2.0  # Seconds to wait before restart attempt
CAMERA_WARMUP_FRAMES = 10   # Skip this many frames on startup for warmup

class VideoServer:
    def __init__(self):
        self.running = True
        self.pico_ip = None
        self.should_stream = False
        self.zed = None
        self.cap = None
        self.grab_fail_count = 0
        self.camera_restart_count = 0
        self.warmup_frames_remaining = CAMERA_WARMUP_FRAMES
        
    def start_command_server(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(('0.0.0.0', COMMAND_PORT))
        server.listen(5)
        print(f"[Video] Command server on port {COMMAND_PORT}")
        
        while self.running:
            try:
                server.settimeout(1.0)
                client, addr = server.accept()
                print(f"[Video] PICO connected from {addr}")
                self.pico_ip = addr[0]
                threading.Thread(target=self.handle_command, args=(client,), daemon=True).start()
            except socket.timeout:
                continue
            except:
                if self.running:
                    pass
    
    def handle_command(self, client):
        try:
            client.settimeout(120.0)
            while self.running:
                header = client.recv(4)
                if not header or len(header) < 4:
                    print(f"[Video] No header received, disconnecting")
                    break
                
                msg_len = struct.unpack('>I', header)[0]
                if msg_len > 10000:
                    msg_len = struct.unpack('<I', header)[0]
                
                print(f"[Video] Receiving message, len={msg_len}")
                
                data = b''
                while len(data) < msg_len:
                    chunk = client.recv(min(4096, msg_len - len(data)))
                    if not chunk:
                        break
                    data += chunk
                
                print(f"[Video] Received {len(data)} bytes: {data[:50]}...")
                
                if b'192.168.' in data:
                    try:
                        ip_start = data.find(b'192.168.')
                        ip_bytes = bytearray()
                        for i in range(ip_start, min(ip_start + 15, len(data))):
                            c = data[i:i+1]
                            if c.isdigit() or c == b'.':
                                ip_bytes.extend(c)
                            else:
                                break
                        self.pico_ip = ip_bytes.decode()
                        print(f"[Video] Extracted PICO IP: {self.pico_ip}")
                    except:
                        pass
                
                if b'OPEN_CAMERA' in data:
                    print(f"[Video] Got OPEN_CAMERA from {self.pico_ip}")
                    client.send(struct.pack('<I', 0))
                    self.should_stream = True
                    threading.Thread(target=self.stream_video, daemon=True).start()
                else:
                    print(f"[Video] Unknown command, sending OK")
                    client.send(struct.pack('<I', 0))
        except Exception as e:
            print(f"[Video] Command handler error: {e}")
        finally:
            print("[Video] Command client disconnected")
            client.close()
    
    def reset_usb_camera(self):
        """Try to reset USB camera device without physical unplug"""
        print("[Video] Attempting USB camera reset...")
        
        # Method 1: Try usbreset command (if available)
        try:
            # Find ZED device
            result = subprocess.run(['lsusb'], capture_output=True, text=True)
            for line in result.stdout.split('\n'):
                if 'ZED' in line or '2b03:f681' in line or '2b03:f582' in line:
                    # Extract bus and device
                    parts = line.split()
                    if len(parts) >= 4:
                        bus = parts[1]
                        dev = parts[3].rstrip(':')
                        dev_path = f'/dev/bus/usb/{bus}/{dev}'
                        print(f"[Video] Found ZED at {dev_path}, resetting...")
                        subprocess.run(['sudo', 'usbreset', dev_path], timeout=5)
                        time.sleep(2)
                        return True
        except Exception as e:
            print(f"[Video] usbreset failed: {e}")
        
        # Method 2: Unbind/rebind USB device via sysfs
        try:
            # Find the USB device path
            result = subprocess.run(
                ['bash', '-c', 'ls /sys/bus/usb/devices/*/product 2>/dev/null | xargs -I{} sh -c \'echo "$(dirname {}): $(cat {})"\''],
                capture_output=True, text=True, timeout=5
            )
            for line in result.stdout.split('\n'):
                if 'ZED' in line:
                    usb_path = line.split(':')[0].split('/')[-1]
                    print(f"[Video] Found ZED USB device: {usb_path}")
                    # Unbind
                    subprocess.run(['sudo', 'bash', '-c', f'echo {usb_path} > /sys/bus/usb/drivers/usb/unbind'], timeout=5)
                    time.sleep(1)
                    # Rebind
                    subprocess.run(['sudo', 'bash', '-c', f'echo {usb_path} > /sys/bus/usb/drivers/usb/bind'], timeout=5)
                    time.sleep(2)
                    print("[Video] USB rebind completed")
                    return True
        except Exception as e:
            print(f"[Video] USB rebind failed: {e}")
        
        return False
    
    def init_camera(self, try_usb_reset=True):
        # Reset warmup counter
        self.warmup_frames_remaining = CAMERA_WARMUP_FRAMES
        
        if USE_ZED_SDK:
            self.zed = sl.Camera()
            init_params = sl.InitParameters()
            init_params.camera_resolution = sl.RESOLUTION.HD720
            init_params.camera_fps = FRAMERATE
            init_params.depth_mode = sl.DEPTH_MODE.NONE
            
            result = self.zed.open(init_params)
            if result == sl.ERROR_CODE.SUCCESS:
                # Warmup: grab a few frames to stabilize
                print("[Video] ZED camera initialized, warming up...")
                for i in range(CAMERA_WARMUP_FRAMES):
                    self.zed.grab()
                    time.sleep(0.05)
                print(f"[Video] ZED ready (skipped {CAMERA_WARMUP_FRAMES} warmup frames)")
                return True
            else:
                print(f"[Video] ZED open failed: {result}")
                self.zed = None
                
                # Try USB reset if first attempt fails
                if try_usb_reset:
                    print("[Video] Trying USB reset...")
                    if self.reset_usb_camera():
                        time.sleep(2)
                        return self.init_camera(try_usb_reset=False)  # Retry without another reset
                
                print("[Video] ZED SDK failed, trying OpenCV")
        
        self.cap = cv2.VideoCapture(0)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)
            # Warmup for OpenCV too
            print("[Video] OpenCV camera initialized, warming up...")
            for i in range(CAMERA_WARMUP_FRAMES):
                self.cap.read()
                time.sleep(0.05)
            print("[Video] OpenCV camera ready")
            return True
        
        print("[Video] No camera available!")
        return False
    
    def get_frame(self):
        if self.zed:
            result = self.zed.grab()
            if result == sl.ERROR_CODE.SUCCESS:
                self.grab_fail_count = 0  # Reset on success
                image = sl.Mat()
                self.zed.retrieve_image(image, sl.VIEW.SIDE_BY_SIDE, sl.MEM.CPU)
                frame = image.get_data()
                if frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                return frame
            else:
                self.grab_fail_count += 1
                if self.grab_fail_count >= CAMERA_FAIL_THRESHOLD:
                    print(f"[Video] Camera grab failed {self.grab_fail_count} times, restarting...")
                    self.restart_camera()
        elif self.cap:
            ret, frame = self.cap.read()
            if ret:
                self.grab_fail_count = 0
                return frame
            else:
                self.grab_fail_count += 1
                if self.grab_fail_count >= CAMERA_FAIL_THRESHOLD:
                    print(f"[Video] Camera read failed {self.grab_fail_count} times, restarting...")
                    self.restart_camera()
        return None
    
    def restart_camera(self):
        """Attempt to restart the camera after failures"""
        self.camera_restart_count += 1
        print(f"[Video] Camera restart attempt #{self.camera_restart_count}")
        
        # Close existing camera
        if self.zed:
            try:
                self.zed.close()
            except:
                pass
            self.zed = None
        
        if self.cap:
            try:
                self.cap.release()
            except:
                pass
            self.cap = None
        
        # Wait before restart
        time.sleep(CAMERA_RESTART_DELAY)
        
        # Try USB reset on every 3rd attempt
        try_usb_reset = (self.camera_restart_count % 3 == 0)
        if try_usb_reset:
            print("[Video] Will try USB reset on this attempt")
        
        # Reinitialize
        self.grab_fail_count = 0
        if self.init_camera(try_usb_reset=try_usb_reset):
            print(f"[Video] Camera restarted successfully (attempt #{self.camera_restart_count})")
        else:
            print(f"[Video] Camera restart failed (attempt #{self.camera_restart_count})")
        return None
    
    def stream_video(self):
        if not self.pico_ip:
            return
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            sock.connect((self.pico_ip, STREAM_PORT))
            sock.settimeout(None)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f"[Video] Streaming to {self.pico_ip}:{STREAM_PORT}")
        except Exception as e:
            print(f"[Video] Connect failed: {e}")
            return
        
        try:
            import av
            self._stream_pyav(sock)
        except ImportError:
            self._stream_ffmpeg(sock)
    
    def _stream_pyav(self, sock):
        import av
        print("[Video] Starting PyAV encoder...")
        output = av.open('pipe:', mode='w', format='h264')
        stream = output.add_stream('libx264', rate=FRAMERATE)
        stream.width = TARGET_WIDTH
        stream.height = TARGET_HEIGHT
        stream.pix_fmt = 'yuv420p'
        stream.bit_rate = BITRATE
        stream.options = {
            'preset': 'ultrafast', 'tune': 'zerolatency',
            'profile': 'baseline', 'x264-params': 'annexb=1:repeat-headers=1'
        }
        
        frame_count = 0
        packet_count = 0
        print(f"[Video] Encoder ready: {TARGET_WIDTH}x{TARGET_HEIGHT}")
        
        try:
            while self.running and self.should_stream:
                frame_bgr = self.get_frame()
                if frame_bgr is None:
                    time.sleep(0.01)
                    continue
                
                if frame_count == 0:
                    print(f"[Video] First frame: {frame_bgr.shape}")
                
                if frame_bgr.shape[1] != TARGET_WIDTH or frame_bgr.shape[0] != TARGET_HEIGHT:
                    frame_bgr = cv2.resize(frame_bgr, (TARGET_WIDTH, TARGET_HEIGHT))
                
                frame_yuv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YUV_I420)
                av_frame = av.VideoFrame(TARGET_WIDTH, TARGET_HEIGHT, 'yuv420p')
                av_frame.planes[0].update(frame_yuv[:TARGET_HEIGHT, :].tobytes())
                u_start = TARGET_HEIGHT
                u_height = TARGET_HEIGHT // 4
                av_frame.planes[1].update(frame_yuv[u_start:u_start + u_height, :].tobytes())
                av_frame.planes[2].update(frame_yuv[u_start + u_height:u_start + 2*u_height, :].tobytes())
                av_frame.pts = frame_count
                
                for packet in stream.encode(av_frame):
                    packet_data = bytes(packet)
                    sock.sendall(struct.pack('>I', len(packet_data)) + packet_data)
                    packet_count += 1
                
                frame_count += 1
                
                time.sleep(1 / FRAMERATE)
        except BrokenPipeError:
            print("[Video] PICO disconnected (broken pipe)")
        except Exception as e:
            print(f"[Video] Stream error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            sock.close()
            output.close()
            print(f"[Video] Stream ended ({frame_count} frames, {packet_count} packets)")
    
    def _stream_ffmpeg(self, sock):
        ffmpeg_cmd = [
            'ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
            '-s', f'{TARGET_WIDTH}x{TARGET_HEIGHT}', '-r', str(FRAMERATE), '-i', '-',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-preset', 'ultrafast',
            '-tune', 'zerolatency', '-profile:v', 'baseline', '-b:v', str(BITRATE),
            '-g', str(FRAMERATE), '-x264-params', 'annexb=1:repeat-headers=1', '-f', 'h264', '-'
        ]
        
        ffmpeg = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, 
                                   stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        
        def sender():
            buffer = b''
            while self.running and self.should_stream:
                try:
                    chunk = ffmpeg.stdout.read(4096)
                    if not chunk:
                        break
                    buffer += chunk
                    while len(buffer) > 1000:
                        sock.sendall(struct.pack('>I', len(buffer)) + buffer)
                        buffer = b''
                except:
                    break
        
        threading.Thread(target=sender, daemon=True).start()
        
        try:
            while self.running and self.should_stream:
                frame = self.get_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue
                if frame.shape[1] != TARGET_WIDTH or frame.shape[0] != TARGET_HEIGHT:
                    frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
                ffmpeg.stdin.write(frame.tobytes())
                time.sleep(1 / FRAMERATE)
        except:
            pass
        finally:
            ffmpeg.terminate()
            sock.close()
    
    def stop(self):
        self.running = False
        self.should_stream = False
        if self.zed:
            self.zed.close()
        if self.cap:
            self.cap.release()


# ============== Neck Controller ==============
class NeckController:
    def __init__(self, yaw_center, pitch_center, device):
        self.yaw_center = yaw_center
        self.pitch_center = pitch_center
        self.device = device
        self.running = True
        self.port_handler = None
        self.packet_handler = None
        self.connected = False
    
    def connect(self):
        try:
            from dynamixel_sdk import PortHandler, PacketHandler
        except ImportError:
            print("[Neck] dynamixel_sdk not installed")
            return False
        
        self.port_handler = PortHandler(self.device)
        self.packet_handler = PacketHandler(PROTOCOL_VERSION)
        
        if not self.port_handler.openPort():
            print(f"[Neck] Failed to open {self.device}")
            return False
        
        if not self.port_handler.setBaudRate(BAUDRATE):
            print(f"[Neck] Failed to set baudrate")
            return False
        
        for motor_id in [ID_YAW, ID_PITCH]:
            name = "YAW" if motor_id == ID_YAW else "PITCH"
            
            # Reboot motor to clear any errors
            self.packet_handler.reboot(self.port_handler, motor_id)
            time.sleep(0.3)  # Wait for reboot
            
            # Set operating mode (returns result, error)
            result, err = self.packet_handler.write1ByteTxRx(self.port_handler, motor_id, ADDR_OPERATING_MODE, 3)
            if result != 0:
                print(f"[Neck] {name} (ID {motor_id}): Failed to set mode (err={result})")
            
            # Set velocity
            self.packet_handler.write4ByteTxRx(self.port_handler, motor_id, ADDR_PROFILE_VELOCITY, 100)
            
            # Enable torque
            result, err = self.packet_handler.write1ByteTxRx(self.port_handler, motor_id, ADDR_TORQUE_ENABLE, 1)
            if result != 0:
                print(f"[Neck] {name} (ID {motor_id}): Failed to enable torque (err={result})")
            
            # Read current position (returns value, result, error)
            pos, result, err = self.packet_handler.read4ByteTxRx(self.port_handler, motor_id, ADDR_PRESENT_POSITION)
            if result == 0:
                print(f"[Neck] {name} (ID {motor_id}): OK, current pos={pos}")
            else:
                print(f"[Neck] {name} (ID {motor_id}): Failed to read position (err={result})")
        
        print(f"[Neck] Connected (yaw_center={self.yaw_center}, pitch_center={self.pitch_center})")
        self.connected = True
        return True
    
    def set_angles(self, yaw_rad, pitch_rad):
        if not self.connected:
            return
        
        positions_per_rad = 4096 / (2 * math.pi)
        yaw_pos = max(YAW_MIN, min(YAW_MAX, 
                      self.yaw_center + int(yaw_rad * positions_per_rad)))
        # Invert pitch direction (negative pitch_rad = look down = motor position increases)
        pitch_pos = max(PITCH_MIN, min(PITCH_MAX,
                        self.pitch_center - int(pitch_rad * positions_per_rad)))
        
        # Write to YAW motor with error checking
        yaw_result, yaw_err = self.packet_handler.write4ByteTxRx(self.port_handler, ID_YAW, ADDR_GOAL_POSITION, yaw_pos)
        if yaw_result != 0:
            print(f"[Neck] YAW write error: result={yaw_result}, err={yaw_err}")
            self._check_motor_error(ID_YAW, "YAW")
        
        # Write to PITCH motor with error checking
        pitch_result, pitch_err = self.packet_handler.write4ByteTxRx(self.port_handler, ID_PITCH, ADDR_GOAL_POSITION, pitch_pos)
        if pitch_result != 0:
            print(f"[Neck] PITCH write error: result={pitch_result}, err={pitch_err}")
            self._check_motor_error(ID_PITCH, "PITCH")
    
    def _check_motor_error(self, motor_id, name):
        """Check and print hardware error status for a motor, attempt recovery"""
        hw_err, result, _ = self.packet_handler.read1ByteTxRx(self.port_handler, motor_id, ADDR_HARDWARE_ERROR)
        if result == 0:
            if hw_err != 0:
                errors = []
                if hw_err & 0x01: errors.append("Input Voltage")
                if hw_err & 0x04: errors.append("Overheating")
                if hw_err & 0x08: errors.append("Motor Encoder")
                if hw_err & 0x10: errors.append("Electrical Shock")
                if hw_err & 0x20: errors.append("Overload")
                print(f"[Neck] {name} HARDWARE ERROR: 0x{hw_err:02X} ({', '.join(errors) if errors else 'unknown'})")
                print(f"[Neck] {name} attempting auto-recovery...")
                self._recover_motor(motor_id, name)
            else:
                print(f"[Neck] {name} no hardware error, check connection")
        else:
            print(f"[Neck] {name} cannot read error status (motor disconnected?)")
            print(f"[Neck] {name} attempting auto-recovery...")
            self._recover_motor(motor_id, name)
    
    def _recover_motor(self, motor_id, name):
        """Attempt to recover a motor by rebooting it"""
        try:
            print(f"[Neck] Rebooting {name}...")
            self.packet_handler.reboot(self.port_handler, motor_id)
            time.sleep(0.5)  # Wait for reboot
            
            # Re-configure motor
            self.packet_handler.write1ByteTxRx(self.port_handler, motor_id, ADDR_OPERATING_MODE, 3)
            self.packet_handler.write4ByteTxRx(self.port_handler, motor_id, ADDR_PROFILE_VELOCITY, 100)
            result, _ = self.packet_handler.write1ByteTxRx(self.port_handler, motor_id, ADDR_TORQUE_ENABLE, 1)
            
            if result == 0:
                print(f"[Neck] {name} recovered successfully!")
            else:
                print(f"[Neck] {name} recovery failed (result={result})")
        except Exception as e:
            print(f"[Neck] {name} recovery exception: {e}")
    
    def center(self):
        if self.connected:
            self.packet_handler.write4ByteTxRx(self.port_handler, ID_YAW, ADDR_GOAL_POSITION, self.yaw_center)
            self.packet_handler.write4ByteTxRx(self.port_handler, ID_PITCH, ADDR_GOAL_POSITION, self.pitch_center)
    
    def run_redis_loop(self, redis_host):
        try:
            import redis
        except ImportError:
            print("[Neck] redis not installed, run: pip3 install redis")
            return
        
        r = redis.Redis(host=redis_host, port=6379, decode_responses=False, socket_timeout=2.0)
        try:
            r.ping()
            print(f"[Neck] Connected to Redis at {redis_host}")
        except:
            print(f"[Neck] Cannot connect to Redis at {redis_host}")
            return
        
        self.center()
        last_yaw, last_pitch = 0.0, 0.0
        
        # Redis watchdog
        last_redis_success = time.time()
        redis_fail_logged = False
        
        while self.running:
            try:
                # Watchdog: ping Redis to verify connection
                r.ping()
                last_redis_success = time.time()
                redis_fail_logged = False
                
                data = r.get('action_neck_unitree_g1_with_hands')
                
                if data:
                    neck_data = json.loads(data)
                    if isinstance(neck_data, (list, tuple)) and len(neck_data) >= 2:
                        yaw, pitch = float(neck_data[0]), float(neck_data[1])
                        if abs(yaw - last_yaw) > 0.01 or abs(pitch - last_pitch) > 0.01:
                            self.set_angles(yaw, pitch)
                            last_yaw, last_pitch = yaw, pitch
                time.sleep(0.02)
                
            except (redis.ConnectionError, redis.TimeoutError) as e:
                # Redis connection lost
                if not redis_fail_logged:
                    print(f"[Neck] Redis connection lost: {e}")
                    redis_fail_logged = True
                
                # Check if timeout exceeded
                if time.time() - last_redis_success > REDIS_WATCHDOG_TIMEOUT:
                    print(f"[Neck] Redis unreachable for {REDIS_WATCHDOG_TIMEOUT}s - shutting down (cable disconnected?)")
                    self.running = False
                    # Signal global shutdown
                    os.kill(os.getpid(), signal.SIGTERM)
                    break
                
                time.sleep(0.1)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"[Neck] Error: {e}")
                time.sleep(0.1)
    
    def stop(self):
        self.running = False
        if self.connected:
            self.packet_handler.write1ByteTxRx(self.port_handler, ID_YAW, ADDR_TORQUE_ENABLE, 0)
            self.packet_handler.write1ByteTxRx(self.port_handler, ID_PITCH, ADDR_TORQUE_ENABLE, 0)
            self.port_handler.closePort()


# ============== Main ==============
def main():
    parser = argparse.ArgumentParser(description='Robot Peripherals for TWIST2')
    parser.add_argument('--redis', default='192.168.50.164', help='Redis server IP')
    parser.add_argument('--yaw_center', type=int, default=DEFAULT_YAW_CENTER)
    parser.add_argument('--pitch_center', type=int, default=DEFAULT_PITCH_CENTER)
    parser.add_argument('--device', default=DEVICENAME, help='Dynamixel device')
    parser.add_argument('--no_video', action='store_true', help='Disable video')
    parser.add_argument('--no_neck', action='store_true', help='Disable neck')
    args = parser.parse_args()
    
    # Setup logging (cleanup old logs)
    setup_logging()
    
    print("=" * 50)
    print("🤖 TWIST2 Robot Peripherals")
    print("=" * 50)
    print(f"   Redis:        {args.redis}")
    print(f"   Video:        {'disabled' if args.no_video else 'enabled'}")
    print(f"   Neck:         {'disabled' if args.no_neck else 'enabled'}")
    if not args.no_neck:
        print(f"   Yaw center:   {args.yaw_center}")
        print(f"   Pitch center: {args.pitch_center}")
    print(f"   Watchdog:     {REDIS_WATCHDOG_TIMEOUT}s timeout")
    print("=" * 50)
    
    video_server = None
    neck_controller = None
    
    # Signal handler for clean shutdown
    def signal_handler(sig, frame):
        print("\n🛑 Shutting down...")
        if video_server:
            video_server.stop()
        if neck_controller:
            neck_controller.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start video server
    if not args.no_video:
        video_server = VideoServer()
        if video_server.init_camera():
            threading.Thread(target=video_server.start_command_server, daemon=True).start()
            print("[Video] Waiting for PICO connection...")
        else:
            video_server = None
    
    # Start neck controller
    if not args.no_neck:
        neck_controller = NeckController(args.yaw_center, args.pitch_center, args.device)
        if neck_controller.connect():
            threading.Thread(target=neck_controller.run_redis_loop, 
                           args=(args.redis,), daemon=True).start()
        else:
            neck_controller = None
    
    # Keep running
    print("\n📡 Running... Press Ctrl+C to stop\n")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    
    signal_handler(None, None)


if __name__ == "__main__":
    main()

