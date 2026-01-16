"""
Hybrid State Machine for PICO + Manus Glove Teleoperation

Manages robot boot sequence and dual-mode finger control:
- Boot sequence: boot → ready → preop
- PICO mode: Trigger-based simple finger control
- Manus mode: Glove-based fine finger control

Button Mapping:
- Unitree START: boot → ready
- Unitree A: ready → preop OR preop → teleop_manus
- PICO Right A: preop → teleop_pico OR (teleop_pico/teleop_manus) → preop
"""

import numpy as np
from robot_control.common.remote_controller import KeyMap


class HybridStateMachine:
    """
    State machine managing robot boot sequence and dual-mode teleoperation
    
    States:
    - "boot" - Initial power up state
    - "ready" - Robot ready, waiting for preop
    - "preop" - Preview mode, waiting for teleop mode selection
    - "teleop_pico" - Active teleop with PICO trigger control
    - "teleop_manus" - Active teleop with Manus glove control
    - "exit" - Shutdown
    
    Button Logic:
    - Unitree START: boot → ready
    - Unitree A: ready → preop OR preop → teleop_manus
    - PICO Right A: preop → teleop_pico OR (teleop_pico/teleop_manus) → preop
    """
    
    def __init__(self):
        self.state = "boot"  # Start in boot state
        self.previous_state = "boot"
        
        # Button edge detection
        self._prev_pico_a = False
        self._prev_unitree_a = False
        self._prev_unitree_start = False
        
        # Hand control state
        self.hand_mode = None  # None, "pico", or "manus"
        
        # PICO hand state (simple trigger-based)
        self.hand_left_position = 0.0
        self.hand_right_position = 0.0
        self.thumb_left_rotation = 0.5
        self.thumb_right_rotation = 0.5
        
        # Hand control parameters
        self.hand_movement_step = 0.05  # 5% per frame when held
        self.thumb_movement_step = 0.03  # 3% per frame for thumb rotation
        
    def update(self, pico_controller_data, unitree_remote_data):
        """
        Update state machine based on button inputs
        
        Args:
            pico_controller_data: Dict from XRobotStreamer.get_controller_data()
            unitree_remote_data: RemoteController instance or dict with button states
        
        Returns:
            Current state string
        """
        self.previous_state = self.state
        
        # Get current button states
        pico_a = pico_controller_data['RightController']['key_one']
        
        # Unitree controller buttons
        if hasattr(unitree_remote_data, 'button'):
            # RemoteController instance
            unitree_a = unitree_remote_data.button[KeyMap.A]
            unitree_start = unitree_remote_data.button[KeyMap.start]
        else:
            # Dict (if read from Redis)
            unitree_a = unitree_remote_data.get('button', [0]*16)[KeyMap.A]
            unitree_start = unitree_remote_data.get('button', [0]*16)[KeyMap.start]
        
        # Detect rising edges (button press, not hold)
        pico_a_pressed = pico_a and not self._prev_pico_a
        unitree_a_pressed = unitree_a and not self._prev_unitree_a
        unitree_start_pressed = unitree_start and not self._prev_unitree_start
        
        # State transitions
        if self.state == "boot":
            if unitree_start_pressed:
                self.state = "ready"
                print("[STATE] boot → ready (Unitree START pressed)")
                
        elif self.state == "ready":
            if unitree_a_pressed:
                self.state = "preop"
                print("[STATE] ready → preop (Unitree A pressed)")
                
        elif self.state == "preop":
            if pico_a_pressed:
                # PICO A → Enter PICO finger mode
                self.state = "teleop_pico"
                self.hand_mode = "pico"
                print("[STATE] preop → teleop_pico (PICO Right A pressed)")
                
            elif unitree_a_pressed:
                # Unitree A → Enter Manus glove mode
                self.state = "teleop_manus"
                self.hand_mode = "manus"
                print("[STATE] preop → teleop_manus (Unitree A pressed)")
                
        elif self.state == "teleop_pico":
            if pico_a_pressed:
                # PICO A → Return to preop
                self.state = "preop"
                self.hand_mode = None
                print("[STATE] teleop_pico → preop (PICO Right A pressed)")
                
        elif self.state == "teleop_manus":
            if pico_a_pressed:
                # PICO A → Return to preop
                self.state = "preop"
                self.hand_mode = None
                print("[STATE] teleop_manus → preop (PICO Right A pressed)")
        
        # Update PICO hand state if in PICO mode
        if self.state == "teleop_pico":
            self._update_pico_hands(pico_controller_data)
        
        # Store previous button states
        self._prev_pico_a = pico_a
        self._prev_unitree_a = unitree_a
        self._prev_unitree_start = unitree_start
        
        return self.state
    
    def _update_pico_hands(self, controller_data):
        """Update PICO trigger-based hand control"""
        left_ctrl = controller_data['LeftController']
        right_ctrl = controller_data['RightController']
        left_x = left_ctrl['key_one']
        left_y = left_ctrl['key_two']
        left_trig = left_ctrl['index_trig']
        right_trig = right_ctrl['index_trig']
        left_grip = left_ctrl['grip']
        right_grip = right_ctrl['grip']
        
        # X + Trigger = Open fingers
        if left_x:
            if left_trig:
                self.hand_left_position = max(0.0, self.hand_left_position - self.hand_movement_step)
            if right_trig:
                self.hand_right_position = max(0.0, self.hand_right_position - self.hand_movement_step)
        
        # Y + Trigger = Close fingers
        if left_y:
            if left_trig:
                self.hand_left_position = min(1.0, self.hand_left_position + self.hand_movement_step)
            if right_trig:
                self.hand_right_position = min(1.0, self.hand_right_position + self.hand_movement_step)
        
        # X + Grip = Thumb outward
        if left_x:
            if left_grip:
                self.thumb_left_rotation = max(0.0, self.thumb_left_rotation - self.thumb_movement_step)
            if right_grip:
                self.thumb_right_rotation = max(0.0, self.thumb_right_rotation - self.thumb_movement_step)
        
        # Y + Grip = Thumb inward
        if left_y:
            if left_grip:
                self.thumb_left_rotation = min(1.0, self.thumb_left_rotation + self.thumb_movement_step)
            if right_grip:
                self.thumb_right_rotation = min(1.0, self.thumb_right_rotation + self.thumb_movement_step)
    
    def is_active(self):
        """Return True if in active teleop (either mode)"""
        return self.state in ["teleop_pico", "teleop_manus"]
    
    def get_hand_mode(self):
        """Return current hand control mode: None, 'pico', or 'manus'"""
        return self.hand_mode
    
    def has_state_changed(self):
        """Check if state changed since last update"""
        return self.state != self.previous_state
    
    def get_pico_hand_state(self):
        """Return PICO hand positions (0=open, 1=closed)"""
        return {
            'left_position': self.hand_left_position,
            'right_position': self.hand_right_position,
            'left_thumb': self.thumb_left_rotation,
            'right_thumb': self.thumb_right_rotation
        }
