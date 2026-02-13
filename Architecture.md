# Virtuals Teleop System Architecture

## Overview

The system is divided into three decoupled components that communicate over network boundaries:

```
┌──────────────┐         ┌─────────────────────┐         ┌──────────────────┐
│              │  WiFi   │                     │  WiFi   │                  │
│     PICO     │ ──────► │ POLICY INFERENCING  │ ──────► │    EXECUTOR      │
│  (VR Device) │ ◄────── │      (PC)           │ ◄────── │    (Robot)       │
│              │         │                     │         │                  │
└──────────────┘         └─────────────────────┘         └──────────────────┘
   User input              Retarget + Policy                Motor control
   Finger feedback         Speed limiting                   Safety layers
   ZED passthrough         State management                 Sensor publishing
```

---

## 1. PICO (VR Device)

Responsible for: User interface, VR tracking input, display feedback, and camera passthrough.

### 1a. Simplified UI

```
pico/
├── ui/
│   ├── hud_manager.py              # Minimal HUD overlay
│   │   ├── class HUDManager
│   │   │   ├── show_status()       # Connection status, robot state
│   │   │   ├── show_finger_force() # Force feedback display
│   │   │   └── show_alerts()       # Safety alerts from Executor
│   │   └── ...
│   └── menu.py                     # Simplified start/stop/config menu
│       ├── class SimpleMenu
│       │   ├── connect()           # One-button connect
│       │   ├── start_teleop()      # Enter teleop mode
│       │   └── stop_teleop()       # Exit teleop mode
│       └── ...
```

### 1b. Simplified Connection to PC

```
pico/
├── connection/
│   ├── pc_link.py                  # Single connection manager
│   │   ├── class PCLink
│   │   │   ├── auto_discover()     # mDNS / broadcast discovery of PC
│   │   │   ├── connect()           # Establish persistent connection
│   │   │   ├── send_tracking()     # Push VR tracking data (head, hands)
│   │   │   ├── receive_feedback()  # Receive finger force, robot state
│   │   │   └── heartbeat()         # Connection health monitoring
│   │   └── ...
│   └── protocol.py                 # Shared message format definitions
│       ├── TrackingMessage         # Head pose, hand poses, button states
│       ├── FeedbackMessage         # Finger forces, robot state, alerts
│       └── ...
```

### 1c. Finger Feedback Display

```
pico/
├── feedback/
│   ├── finger_force_display.py     # Render Inspire force data on HUD
│   │   ├── class FingerForceDisplay
│   │   │   ├── update(force_data)  # Receive force array from PC
│   │   │   ├── render_overlay()    # Draw per-finger force bars
│   │   │   └── alert_threshold()   # Flash warning if force limit hit
│   │   └── ...
```

### 1d. ZED Mini Connection

```
pico/
├── passthrough/
│   ├── zed_receiver.py             # Receive ZED stereo stream from Executor
│   │   ├── class ZEDReceiver
│   │   │   ├── connect(robot_addr) # Connect to ZED stream from robot
│   │   │   ├── decode_frame()      # Decode stereo pair
│   │   │   └── render_passthrough()# Display in VR headset
│   │   └── ...
```

---

## 2. POLICY INFERENCING (PC)

Responsible for: Receiving VR tracking, retargeting to robot joints, running locomotion/teleop policies, applying safety limits, and forwarding joint commands to Executor.

### 2a. Teleop + Policy Pipeline (to be restructured)

```
policy/
├── pipeline/
│   ├── teleop_server.py            # Main entry point
│   │   ├── class TeleopServer
│   │   │   ├── __init__()          # Load policies, connect to Pico & Executor
│   │   │   ├── run()               # Main loop
│   │   │   └── shutdown()          # Graceful cleanup
│   │   └── ...
│   │
│   ├── input_stage.py              # Stage 1: Receive + validate VR data
│   │   ├── class InputStage
│   │   │   ├── receive()           # Get tracking from Pico
│   │   │   ├── validate()          # Reject spikes, check bounds
│   │   │   └── get_tracking()      # Return clean VR data
│   │   └── ...
│   │
│   ├── retarget_stage.py           # Stage 2: VR poses → robot joint targets
│   │   ├── class RetargetStage
│   │   │   ├── retarget_upper()    # Arms, hands, waist from VR
│   │   │   ├── retarget_neck()     # Head tracking → neck joints
│   │   │   └── get_joint_targets() # Return 29-DOF target
│   │   └── ...
│   │
│   ├── loco_stage.py               # Stage 3: Locomotion policy
│   │   ├── class LocoStage
│   │   │   ├── compute()           # Run GearWBC or TWIST2 policy
│   │   │   ├── blend()             # Merge upper body + lower body
│   │   │   └── get_action()        # Return full-body joint targets
│   │   └── ...
│   │
│   ├── safety_stage.py             # Stage 4: Pre-send safety
│   │   ├── class SafetyStage
│   │   │   ├── speed_limit()       # See 2c below
│   │   │   ├── joint_limit()       # Clamp to safe ranges
│   │   │   └── get_safe_action()   # Return validated action
│   │   └── ...
│   │
│   └── state_machine.py            # Teleop state management
│       ├── class TeleopStateMachine
│       │   ├── states: idle, preview, teleop_full, teleop_loco, paused
│       │   ├── transition()        # Handle state changes
│       │   └── current_state()     # Get current state
│       └── ...
```

### 2b. Finger Force Feedback Channel

```
policy/
├── feedback/
│   ├── force_relay.py              # Read Inspire force → push to Pico
│   │   ├── class ForceRelay
│   │   │   ├── read_forces()       # Poll Inspire hand force sensors
│   │   │   ├── package()           # Format into FeedbackMessage
│   │   │   └── send_to_pico()     # Push over PCLink
│   │   └── ...
```

### 2c. Policy Stability Improvements

```
policy/
├── stability/
│   ├── speed_limiter.py            # Limit effective joint velocity
│   │   ├── class SpeedLimiter
│   │   │   ├── __init__(max_vel_per_joint)
│   │   │   ├── limit(prev_target, new_target, dt)
│   │   │   │   # Clamps per-joint delta to max_vel * dt
│   │   │   │   # Robot moves at its own pace regardless of user speed
│   │   │   └── reset()
│   │   └── ...
│   │
│   ├── lean_compensator.py         # Hip pitch offset logic (current LEAN_OFFSET)
│   │   ├── class LeanCompensator
│   │   │   ├── __init__(lean_close, lean_far)
│   │   │   ├── compute_offset(dof_pos)   # FK-based arm position → offset
│   │   │   ├── pre_policy(dof_pos)       # Subtract offset before policy
│   │   │   └── post_policy(loco_action)  # Add offset after policy
│   │   └── ...
│   │
│   └── standing_stabilizer.py      # Ensure stable standing in any arm pose
│       ├── class StandingStabilizer
│       │   ├── __init__(height_cmd, kp_overrides)
│       │   ├── adjust_for_arm_pose(arm_dof_pos)
│       │   └── get_params()        # Return height_cmd, kps, rpy_cmd
│       └── ...
```

---

## 3. EXECUTOR (Robot)

Responsible for: Receiving joint commands over wireless, executing them through the motor controller with layered safety, publishing sensor data back up the chain.

### 3a. Safety — 5-Tier Operation

```
executor/
├── safety/
│   ├── safety_manager.py           # Central safety state machine
│   │   ├── class SafetyManager
│   │   │   ├── tier: NORMAL → INTERPOLATE → STALL → KNEEL → KILL
│   │   │   │
│   │   │   │   NORMAL:      Execute commands as received
│   │   │   │   INTERPOLATE: Commands arriving late/sparse,
│   │   │   │                interpolate between last known targets
│   │   │   │   STALL:       Commands stopped, hold current position
│   │   │   │   KNEEL:       Controlled descent to kneeling pose
│   │   │   │   KILL:        Emergency — cut motor power immediately
│   │   │   │
│   │   │   ├── update(cmd_age, imu_data, motor_state)
│   │   │   │   # Evaluate conditions and transition tiers
│   │   │   ├── get_tier()
│   │   │   └── execute(target_dof_pos, tier)
│   │   │       # Apply tier-appropriate behavior to command
│   │   └── ...
│   │
│   ├── tier_triggers.py            # Conditions for tier transitions
│   │   ├── NORMAL → INTERPOLATE:  cmd_age > 50ms
│   │   ├── INTERPOLATE → STALL:   cmd_age > 200ms
│   │   ├── STALL → KNEEL:         cmd_age > 2s  OR  IMU anomaly
│   │   ├── ANY → KILL:            hardware fault, overcurrent, user e-stop
│   │   └── recovery: KILL requires manual reset
│   │       KNEEL → STALL:         commands resume
│   │       STALL → INTERPOLATE:   commands resume + stable for 500ms
│   │       INTERPOLATE → NORMAL:  cmd_age < 20ms sustained
│   │
│   └── kneel_controller.py        # Safe descent sequence
│       ├── class KneelController
│       │   ├── compute_trajectory()  # Smooth path to kneeling pose
│       │   ├── step()                # Next waypoint in kneel sequence
│       │   └── is_complete()         # Reached safe ground pose
│       └── ...
```

### 3b. ZED Mini Publishing

```
executor/
├── sensors/
│   ├── zed_publisher.py            # Capture + stream ZED stereo to PC/Pico
│   │   ├── class ZEDPublisher
│   │   │   ├── __init__(resolution, fps, compression)
│   │   │   ├── capture()           # Grab stereo frame
│   │   │   ├── encode()            # Compress for network
│   │   │   └── publish()           # Stream to Pico (for passthrough)
│   │   └── ...
│   │
│   └── imu_publisher.py            # Robot IMU + joint state feedback
│       ├── class IMUPublisher
│       │   ├── read_state()        # IMU quaternion, angular vel, joint pos/vel
│       │   └── publish()           # Send to Policy Inferencing PC
│       └── ...
```

### 3c. Neck Controller (Optional)

```
executor/
├── peripherals/
│   ├── neck_controller.py          # Neck servo control from head tracking
│   │   ├── class NeckController
│   │   │   ├── __init__(calibration)
│   │   │   ├── update(head_rpy)        # Receive head orientation target
│   │   │   ├── filter(raw_cmd)         # Spike rejection + low-pass
│   │   │   ├── send_to_servos()        # Write to neck hardware
│   │   │   └── is_available()          # Check if neck hardware present
│   │   └── ...
```

### 3d. Command Receiver + Motor Interface

```
executor/
├── core/
│   ├── command_receiver.py         # Receive joint targets from PC
│   │   ├── class CommandReceiver
│   │   │   ├── listen()            # UDP/TCP listener for joint commands
│   │   │   ├── get_latest()        # Return most recent command + age
│   │   │   └── cmd_age()           # Time since last valid command
│   │   └── ...
│   │
│   ├── motor_interface.py          # Unitree low-level motor control
│   │   ├── class MotorInterface
│   │   │   ├── __init__(robot_type)    # G1, T1, etc.
│   │   │   ├── send_targets(pos, kp, kd)
│   │   │   ├── read_state()            # Joint pos, vel, torque
│   │   │   └── emergency_stop()        # Cut all motors
│   │   └── ...
│   │
│   └── executor_main.py            # Main loop
│       ├── class Executor
│       │   ├── __init__()          # Init all subsystems
│       │   ├── run()               # Main loop:
│       │   │   #   1. command_receiver.get_latest()
│       │   │   #   2. safety_manager.update()
│       │   │   #   3. safety_manager.execute()
│       │   │   #   4. motor_interface.send_targets()
│       │   │   #   5. imu_publisher.publish()
│       │   │   #   6. zed_publisher.publish()
│       │   │   #   7. neck_controller.update() (if available)
│       │   └── shutdown()
│       └── ...
```

---

## Communication Flow

```
                    PICO                          PC                           ROBOT
                 ┌─────────┐               ┌──────────────┐              ┌─────────────┐
                 │ Tracking │──── VR ──────►│ Input Stage  │              │             │
                 │  Data    │   Poses       │      ↓       │              │             │
                 │         │               │ Retarget     │              │             │
                 │         │               │      ↓       │              │             │
                 │         │               │ Loco Policy  │              │             │
                 │         │               │      ↓       │              │             │
                 │         │               │ Safety Stage │── Joints ──►│ Cmd Receiver│
                 │         │               │              │   + PD      │      ↓      │
                 │         │               │              │             │ Safety Mgr  │
                 │         │               │              │             │      ↓      │
                 │  Force  │◄── Finger ────│ Force Relay  │◄── State ──│ Motor Iface │
                 │ Display │   Forces      │              │   + IMU    │      ↓      │
                 │         │               │              │             │ Sensors     │
                 │  ZED    │◄── Stereo ────│──────────────│◄── Video ──│ ZED Publish │
                 │Passthru │   Stream      │              │             │             │
                 └─────────┘               └──────────────┘              └─────────────┘
```

---

## Transport Layer (TBD)

| Link | Current | Target |
|------|---------|--------|
| Pico → PC | Redis / USB tethered | WiFi UDP (low-latency) |
| PC → Robot | Redis over SSH tunnel | WiFi UDP + heartbeat |
| Robot → PC | Redis over SSH tunnel | WiFi UDP (state feedback) |
| Robot → Pico | N/A | WiFi UDP (ZED stream) |

---

## Notes

- **Pico** is pure I/O — no policy logic runs on it
- **Policy Inferencing** is the brain — all retargeting, policy, and state management lives here
- **Executor** is pure actuation + safety — it doesn't know what a "teleop" is, it just receives joint targets and executes them safely
- The 5-tier safety in Executor means the robot is safe even if PC crashes or WiFi drops
- Speed limiter in Policy means the robot moves smoothly even if the user flails

