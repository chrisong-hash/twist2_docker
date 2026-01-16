# TWIST2 Hybrid Teleoperation System

Integration of PICO controller trigger-based control with Manus glove fine finger control.

## Overview

This system allows switching between two finger control modes:
- **PICO Mode**: Simple trigger-based finger control (good for general manipulation)
- **Manus Mode**: Fine glove-based finger control (good for precise tasks)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  ROBOT BOOT SEQUENCE                         │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│                    ┌──────────────┐                          │
│                    │ POWER UP     │                          │
│                    └──────┬───────┘                          │
│                           │                                  │
│                    Unitree START                             │
│                           │                                  │
│                           ▼                                  │
│                    ┌──────────────┐                          │
│                    │ READY STATE  │                          │
│                    └──────┬───────┘                          │
│                           │                                  │
│                    Unitree A                                 │
│                           │                                  │
│                           ▼                                  │
│                    ┌──────────────┐                          │
│                    │ PREOP MODE   │◄─────────┐              │
│                    │ (preview)    │          │              │
│                    └──────┬───────┘          │              │
│                           │                  │              │
│              ┌────────────┴────────────┐     │              │
│              │                         │     │              │
│        PICO Right A              Unitree A   │              │
│              │                         │     │              │
│              ▼                         ▼     │              │
│    ┌──────────────────┐      ┌──────────────────┐          │
│    │ PICO FINGER MODE │      │ MANUS GLOVE MODE │          │
│    │ - Body: PICO     │      │ - Body: PICO     │          │
│    │ - Hands: Triggers│      │ - Hands: Gloves  │          │
│    └────────┬─────────┘      └────────┬─────────┘          │
│             │                         │                     │
│      PICO Right A              PICO Right A                 │
│             │                         │                     │
│             └────────────┬────────────┘                     │
│                          │                                  │
│                          └──────────────────────────────────┘
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

## Button Mapping

| State | Button | Action |
|-------|--------|--------|
| Power Up | Unitree START | → Ready State |
| Ready | Unitree A | → Preop Mode |
| Preop | PICO Right A | → PICO Finger Mode |
| Preop | Unitree A | → Manus Glove Mode |
| PICO Finger Mode | PICO Right A | → Preop Mode |
| Manus Glove Mode | PICO Right A | → Preop Mode |

### PICO Finger Control (when in PICO mode)
- **Left X + Left Trigger**: Open left fingers
- **Left X + Right Trigger**: Open right fingers
- **Left Y + Left Trigger**: Close left fingers
- **Left Y + Right Trigger**: Close right fingers
- **Left X + Left Grip**: Left thumb outward
- **Left X + Right Grip**: Right thumb outward
- **Left Y + Left Grip**: Left thumb inward
- **Left Y + Right Grip**: Right thumb inward

## Usage

### Terminal 1: Start Low-Level Controller

```bash
cd /home/chris/CodeSpace/twist2_docker
docker exec -it twist2 bash
cd /workspace/twist2
bash sim2real.sh
```

This will:
- Initialize robot connection
- Start policy server
- Publish Unitree remote controller state to Redis

### Terminal 2: Start Hybrid Teleop

```bash
docker exec -it twist2 bash
cd /workspace/twist2
bash teleop_hybrid.sh
```

### Boot Sequence

1. **System starts in BOOT state**
   - Waiting for Unitree START button

2. **Press Unitree START**
   - System enters READY state
   - Robot initializes

3. **Press Unitree A**
   - System enters PREOP (preview) mode
   - Robot moves to default pose
   - Waiting for mode selection

4. **Select Mode:**
   - **Press PICO Right A** → Enter PICO finger mode
   - **Press Unitree A** → Enter Manus glove mode

5. **Return to Preview:**
   - **Press PICO Right A** (from either mode) → Return to PREOP

## File Structure

```
twist2_docker/twist2/deploy_real/
├── xrobot_teleop_hybrid.py         # Main hybrid teleop script
├── hybrid_state_machine.py         # State machine implementation
├── xsens_manus_integration/        # Xsens/Manus components
│   ├── __init__.py
│   ├── xsens_streamer.py          # UDP listener for Xsens data
│   ├── udp_listener.py            # Packet decoder
│   ├── inspire_hand_streamer_v2.py # Manus → Inspire mapper
│   ├── offsets.json               # Xsens calibration
│   ├── calibration_open_palm.json
│   └── calibration_closed_fist.json
├── server_low_level_g1_real.py    # Modified to publish remote state
└── robot_control/
    └── inspire_hand_wrapper.py    # Inspire hand controller
```

## Configuration

Edit `teleop_hybrid.sh` to change:

```bash
actual_human_height=1.80           # Your height in meters
inspire_left_ip="192.168.123.210"  # Left Inspire hand IP
inspire_right_ip="192.168.123.211" # Right Inspire hand IP
xsens_port=9763                    # Xsens UDP port
```

## Troubleshooting

### PICO Controller Not Detected
- Check XRoboToolkit PC Service is running
- Verify PICO is connected and streaming
- Check network connectivity

### Manus Gloves Not Working
- Verify Xsens suit is streaming on port 9763
- Check Manus gloves are calibrated
- Test with standalone `xsens_teleop` first

### Unitree Remote Not Responding
- Ensure `server_low_level_g1_real.py` is running
- Check Redis connection: `redis-cli get unitree_remote_state`
- Verify robot is in dev mode (L2+R2)

### Inspire Hands Not Moving
- Check IP addresses are correct
- Verify hands are powered and connected
- Test connection: `ping 192.168.123.210`

## Network Setup

### For Xsens/Manus Gloves:
- Xsens suit must stream UDP packets to port 9763
- Configure Xsens MVN to stream to your PC's IP
- Firewall must allow UDP on port 9763

### For Robot:
- Robot: `192.168.123.164`
- PC: `192.168.123.222`
- Netmask: `255.255.255.0`

## Testing Without Hardware

### Test State Machine Only:
```python
# In Python shell
from hybrid_state_machine import HybridStateMachine
sm = HybridStateMachine()
# Simulate button presses
```

### Test with Sim2Sim:
```bash
# Terminal 1
bash sim2sim.sh

# Terminal 2
bash teleop_hybrid.sh --enable_manus
```

## Development

### Adding New States:
Edit `hybrid_state_machine.py` and add to state transitions.

### Changing Hand Mapping:
Edit `xsens_manus_integration/inspire_hand_streamer_v2.py`
- Modify `TeleVisionStyleMapper` class
- Adjust calibration values

### Debugging:
- Enable FPS monitoring: `--measure_fps 1`
- Check Redis keys: `redis-cli keys '*'`
- Monitor state: Watch console output for state transitions

## Credits

- TWIST2: Amazon Far AI Research
- GMR: General Motion Retargeting
- Xsens Integration: Custom implementation
- Manus Glove Mapping: TeleVision-style approach
