"""
Inspire Hand RH56DFTP Controller
Modbus TCP communication with Inspire dexterous hands
Supports async mode for non-blocking operation
"""
import socket
import struct
import numpy as np
import time
import threading
import queue


class InspireHandController:
    """Controller for a single Inspire Hand via Modbus TCP"""
    
    # Register addresses (from RH56DFTP User Manual V1.0.0)
    SET_ANGLE_REG = 1486      # Write angles (ANGLE_SET)
    SET_FORCE_REG = 1498      # Write force threshold (FORCE_SET) - limits closing force
    SET_SPEED_REG = 1522      # Write speed (SPEED_SET)
    DEFAULT_FORCE_REG = 1044  # Power-on default force threshold (DEFAULT_FORCE_SET)
    GET_POSITION_REG = 1534   # Read position (POS_ACT)
    GET_ANGLE_REG = 1546      # Read angles (ANGLE_ACT)
    GET_FORCE_REG = 1582      # Read force (FORCE_ACT)
    GET_CURRENT_REG = 1594    # Read current (CURRENT)
    GET_ERROR_REG = 1606      # Read errors (ERROR)
    GET_STATUS_REG = 1612     # Read status (STATUS)
    GET_TEMP_REG = 1618       # Read temperature (TEMP)
    RESET_ERROR_REG = 1004    # Reset errors (CLEAR_ERROR)
    
    # Angle range (0=open, 2000=closed for fingers)
    ANGLE_MIN = 0
    ANGLE_MAX = 2000
    
    # Modbus constants
    PORT = 6000
    UNIT_ID = 1
    TIMEOUT = 2.0
    
    # DOF names
    DOF_NAMES = ['Little', 'Ring', 'Middle', 'Index', 'Thumb_Bend', 'Thumb_Rotate']
    NUM_DOFS = 6
    
    # Async constants
    MAX_QUEUE_SIZE = 10
    
    def __init__(self, ip, port=PORT, timeout=TIMEOUT, async_mode=True, verbose=False):
        """
        Initialize Inspire Hand controller
        
        Args:
            ip: IP address of the hand
            port: Modbus TCP port (default 6000)
            timeout: Socket timeout in seconds
            async_mode: If True, commands are queued and sent in background thread
            verbose: If True, print all errors (default: only first error)
        """
        self.ip = ip
        self.port = port
        self.timeout = timeout
        self.async_mode = async_mode
        self.verbose = verbose
        self.sock = None
        self.transaction_id = 0
        self._connect_error_printed = False
        
        # Async mode
        self.command_queue = queue.Queue(maxsize=self.MAX_QUEUE_SIZE)
        self.worker_thread = None
        self.running = False
        
        if self.async_mode:
            self._start_worker_thread()
    
    def _start_worker_thread(self):
        """Start background worker thread for async mode"""
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_thread, daemon=True)
        self.worker_thread.start()
    
    def _worker_thread(self):
        """Background worker that processes queued commands"""
        while self.running:
            try:
                command = self.command_queue.get(timeout=0.1)
                if command is None:  # Sentinel to stop thread
                    break
                
                func_code, reg_addr, data_to_send = command
                
                # Reconnect for each command (Inspire hands expect this)
                if self._connect():
                    response = self._send_modbus_command_raw(func_code, reg_addr, data_to_send)
                    if response:
                        pass  # Command sent successfully
                
                self.command_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                if self.verbose:
                    print(f"[ASYNC ERROR] Hand {self.ip}: {e}")
                self.disconnect()
    
    def _connect(self):
        """Establish TCP connection"""
        try:
            if self.sock:
                self.sock.close()
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(self.timeout)
            self.sock.connect((self.ip, self.port))
            self._connect_error_printed = False  # Reset on successful connect
            return True
        except socket.error as e:
            if self.verbose or not self._connect_error_printed:
                print(f"[ERROR] Failed to connect to {self.ip}:{self.port} - {e}")
                self._connect_error_printed = True
            self.sock = None
            return False
    
    def disconnect(self):
        """Close TCP connection"""
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
            self.sock = None
    
    def _send_modbus_command(self, function_code, start_address, data):
        """
        Send Modbus command (async or blocking)
        
        Args:
            function_code: Modbus function code (0x03, 0x06, 0x10)
            start_address: Register start address
            data: Data to send (int for FC06, list/array for FC10, int count for FC03)
        """
        if self.async_mode:
            # Queue command for background processing
            try:
                if self.command_queue.qsize() >= self.MAX_QUEUE_SIZE:
                    # Drop oldest command if queue full
                    try:
                        self.command_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.command_queue.put((function_code, start_address, data))
                return True
            except Exception as e:
                if self.verbose:
                    print(f"[ERROR] Failed to queue command: {e}")
                return False
        else:
            # Blocking mode - send immediately
            if not self.sock or self.sock.fileno() == -1:
                if not self._connect():
                    return None
            return self._send_modbus_command_raw(function_code, start_address, data)
    
    def _send_modbus_command_raw(self, function_code, start_address, data):
        """
        Send Modbus TCP command (raw, blocking)
        
        Args:
            function_code: Modbus function code
            start_address: Register start address
            data: Data to send
        """
        try:
            self.transaction_id += 1
            if self.transaction_id > 65535:
                self.transaction_id = 1
            
            # Build Modbus TCP message
            if function_code == 0x03:  # Read Holding Registers
                count = data
                pdu = struct.pack(">BHH", function_code, start_address, count)
            elif function_code == 0x06:  # Write Single Register
                value = data
                pdu = struct.pack(">BHH", function_code, start_address, value)
            elif function_code == 0x10:  # Write Multiple Registers
                values = data if isinstance(data, (list, np.ndarray)) else [data]
                count = len(values)
                byte_count = count * 2
                pdu = struct.pack(">BHHB", function_code, start_address, count, byte_count)
                for val in values:
                    pdu += struct.pack(">H", int(val))
            else:
                print(f"[ERROR] Unsupported function code: {function_code}")
                return None
            
            # MBAP Header
            mbap = struct.pack(">HHHB", 
                             self.transaction_id,  # Transaction ID
                             0,                     # Protocol ID (0 for Modbus)
                             len(pdu) + 1,          # Length
                             self.UNIT_ID)          # Unit ID
            
            message = mbap + pdu
            
            # Send and receive
            self.sock.sendall(message)
            response = self.sock.recv(1024)
            
            # Close socket after each command (Inspire hands expect this!)
            if self.sock:
                self.sock.close()
                self.sock = None
            
            if len(response) < 8:
                if self.verbose:
                    print(f"[ERROR] Response too short: {len(response)} bytes")
                return None
            
            # Parse response
            resp_trans_id, resp_proto_id, resp_length, resp_unit_id = struct.unpack(">HHHB", response[:7])
            resp_func_code = response[7]
            
            if resp_func_code & 0x80:
                if self.verbose:
                    error_code = response[8] if len(response) > 8 else 0
                    print(f"[ERROR] Modbus exception: function={resp_func_code & 0x7F}, error={error_code}")
                return None
            
            return response[7:]  # Return PDU only
            
        except socket.timeout:
            if self.verbose:
                print(f"[ERROR] Communication timeout with {self.ip}")
            self.disconnect()
            return None
        except socket.error as e:
            if self.verbose:
                print(f"[ERROR] Socket error with {self.ip}: {e}")
            self.disconnect()
            return None
        except Exception as e:
            if self.verbose:
                print(f"[ERROR] Unexpected error: {e}")
            self.disconnect()
            return None
    
    def set_angles(self, angles):
        """
        Set all 6 DOF angles
        
        Args:
            angles: Array of 6 angles (0-2000 range, 0=open, 2000=closed)
        """
        if len(angles) != self.NUM_DOFS:
            print(f"[ERROR] Expected {self.NUM_DOFS} angles, got {len(angles)}")
            return False
        
        # Clamp and convert to int
        angles_clamped = np.clip(angles, self.ANGLE_MIN, self.ANGLE_MAX).astype(np.int16)
        
        return self._send_modbus_command(0x10, self.SET_ANGLE_REG, angles_clamped.tolist())
    
    def set_position_normalized(self, finger_position, thumb_position=None, thumb_rotation=0.5):
        """
        Set hand position with normalized values
        
        Args:
            finger_position: 0.0 = fingers (DOF 0-3) fully open, 1.0 = fully closed
            thumb_position: 0.0 = thumb (DOF 4) open, 1.0 = closed. If None, follows finger_position
            thumb_rotation: 0.0 = outward, 0.5 = neutral, 1.0 = inward (DOF 5)
        """
        finger_position = np.clip(finger_position, 0.0, 1.0)
        thumb_rotation = np.clip(thumb_rotation, 0.0, 1.0)
        
        # If thumb_position not specified, follow fingers
        if thumb_position is None:
            thumb_position = finger_position
        else:
            thumb_position = np.clip(thumb_position, 0.0, 1.0)
        
        # Map 0.0-1.0 to ANGLE_MAX-ANGLE_MIN (inverted: 0.0=open=2000, 1.0=closed=0)
        finger_angle = self.ANGLE_MAX - (finger_position * (self.ANGLE_MAX - self.ANGLE_MIN))
        angles = np.full(self.NUM_DOFS, finger_angle, dtype=np.int16)
        
        # Thumb bend (DOF 4): separate control with lag support
        thumb_bend_angle = self.ANGLE_MAX - (thumb_position * (self.ANGLE_MAX - self.ANGLE_MIN))
        angles[4] = int(thumb_bend_angle)
        
        # Thumb rotation (DOF 5): 0=outward(2000), 0.5=neutral(1000), 1=inward(0)
        thumb_rot_angle = self.ANGLE_MAX - (thumb_rotation * (self.ANGLE_MAX - self.ANGLE_MIN))
        angles[5] = int(thumb_rot_angle)
        
        return self.set_angles(angles)
    
    def open_hand(self):
        """Open hand completely"""
        return self.set_position_normalized(0.0)
    
    def close_hand(self):
        """Close hand completely"""
        return self.set_position_normalized(1.0)
    
    def read_angles(self):
        """Read current angles from hand"""
        response = self._send_modbus_command(0x03, self.GET_ANGLE_REG, self.NUM_DOFS)
        if response and len(response) >= 2 + self.NUM_DOFS * 2:
            byte_count = response[1]
            angles = []
            for i in range(self.NUM_DOFS):
                offset = 2 + i * 2
                angle = struct.unpack(">H", response[offset:offset+2])[0]
                angles.append(angle)
            return np.array(angles)
        return None
    
    def set_force_threshold(self, force_values):
        """
        Set force threshold for each DOF to limit closing force.
        Lower values = weaker grip, less likely to get stuck.
        
        Args:
            force_values: Array of 6 force thresholds (0-3000 range per manual)
                         DOF order: [little, ring, middle, index, thumb_bend, thumb_rot]
                         Lower = softer grip, Higher = stronger grip
                         Example from manual: 300 = soft grip
                         Recommended: 300-800 for normal use
        """
        if len(force_values) != self.NUM_DOFS:
            print(f"[ERROR] Expected {self.NUM_DOFS} force values, got {len(force_values)}")
            return False
        
        force_clamped = np.clip(force_values, 0, 3000).astype(np.int16)
        return self._send_modbus_command(0x10, self.SET_FORCE_REG, force_clamped.tolist())
    
    def set_force_threshold_all(self, force_value):
        """
        Set same force threshold for all DOFs.
        
        Args:
            force_value: Force threshold (0-3000 range)
                        - 200-400: Very soft grip (safe testing)
                        - 500-800: Medium grip (recommended)
                        - 1000-1500: Strong grip
                        - 3000: Maximum force (may cause stuck fingers)
        """
        force_values = [int(force_value)] * self.NUM_DOFS
        return self.set_force_threshold(force_values)
    
    def read_force(self):
        """Read actual force applied to each finger"""
        response = self._send_modbus_command(0x03, self.GET_FORCE_REG, self.NUM_DOFS)
        if response and len(response) >= 2 + self.NUM_DOFS * 2:
            forces = []
            for i in range(self.NUM_DOFS):
                offset = 2 + i * 2
                force = struct.unpack(">H", response[offset:offset+2])[0]
                forces.append(force)
            return np.array(forces)
        return None
    
    def set_speed(self, speed_values):
        """
        Set speed for each DOF. Lower speed = gentler closing.
        
        Args:
            speed_values: Array of 6 speed values (0-1000)
        """
        if len(speed_values) != self.NUM_DOFS:
            print(f"[ERROR] Expected {self.NUM_DOFS} speed values, got {len(speed_values)}")
            return False
        
        speed_clamped = np.clip(speed_values, 0, 1000).astype(np.int16)
        return self._send_modbus_command(0x10, self.SET_SPEED_REG, speed_clamped.tolist())
    
    def set_speed_all(self, speed_value):
        """Set same speed for all DOFs."""
        speed_values = [int(speed_value)] * self.NUM_DOFS
        return self.set_speed(speed_values)
    
    def stop(self):
        """Stop async worker thread"""
        if self.async_mode and self.running:
            self.running = False
            self.command_queue.put(None)  # Sentinel
            if self.worker_thread:
                self.worker_thread.join(timeout=2.0)
        self.disconnect()


class DualHandController:
    """Controller for both left and right Inspire hands"""
    
    def __init__(self, left_ip, right_ip, timeout=2.0, async_mode=True):
        """
        Initialize dual hand controller
        
        Args:
            left_ip: IP address of left hand
            right_ip: IP address of right hand
            timeout: Socket timeout in seconds
            async_mode: Use async mode for non-blocking operation
        """
        print(f"[INSPIRE] Initializing Inspire hands...")
        print(f"[INSPIRE] Left hand: {left_ip}, Right hand: {right_ip}")
        print(f"[INSPIRE] Async mode: {async_mode}")
        
        self.left_hand = InspireHandController(left_ip, timeout=timeout, async_mode=async_mode)
        self.right_hand = InspireHandController(right_ip, timeout=timeout, async_mode=async_mode)
        
        # Initial connection test
        time.sleep(0.5)  # Give async threads time to start
        
        print("[INSPIRE] ✓ Inspire hands initialized")
    
    def ctrl_dual_hand(self, left_finger_pos, right_finger_pos,
                        left_thumb_pos=None, right_thumb_pos=None,
                        left_thumb_rotation=0.5, right_thumb_rotation=0.5):
        """
        Control both hands with normalized positions
        
        Args:
            left_finger_pos: 0.0-1.0 (fingers: 0.0=open, 1.0=closed)
            right_finger_pos: 0.0-1.0 (fingers: 0.0=open, 1.0=closed)
            left_thumb_pos: 0.0-1.0 (thumb bend: 0.0=open, 1.0=closed). None=follow fingers
            right_thumb_pos: 0.0-1.0 (thumb bend: 0.0=open, 1.0=closed). None=follow fingers
            left_thumb_rotation: 0.0=outward, 0.5=neutral, 1.0=inward
            right_thumb_rotation: 0.0=outward, 0.5=neutral, 1.0=inward
        """
        self.left_hand.set_position_normalized(left_finger_pos, left_thumb_pos, left_thumb_rotation)
        self.right_hand.set_position_normalized(right_finger_pos, right_thumb_pos, right_thumb_rotation)
    
    def open_both(self):
        """Open both hands"""
        self.left_hand.open_hand()
        self.right_hand.open_hand()
    
    def close_both(self):
        """Close both hands"""
        self.left_hand.close_hand()
        self.right_hand.close_hand()
    
    def set_force_limit(self, force_value):
        """
        Set force threshold for both hands to prevent fingers getting stuck.
        
        Args:
            force_value: Force threshold (0-3000 per Inspire manual)
                        - 200-400: Very soft grip (safe for testing)
                        - 500-800: Medium grip (recommended)
                        - 1000-1500: Strong grip
                        - 3000: Maximum force (may cause stuck fingers)
                        
        Note: Manual example uses 300 for index finger soft grip
        """
        print(f"[INSPIRE] Setting force limit to {force_value} on both hands")
        self.left_hand.set_force_threshold_all(force_value)
        self.right_hand.set_force_threshold_all(force_value)
    
    def set_speed(self, speed_value):
        """
        Set closing speed for both hands.
        
        Args:
            speed_value: Speed (0-1000), lower = gentler closing
        """
        print(f"[INSPIRE] Setting speed to {speed_value} on both hands")
        self.left_hand.set_speed_all(speed_value)
        self.right_hand.set_speed_all(speed_value)
    
    def stop(self):
        """Stop both hands"""
        self.left_hand.stop()
        self.right_hand.stop()