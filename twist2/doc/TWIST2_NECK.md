# TWIST2 Neck Control

This document covers the Dynamixel-based neck control system for TWIST2 teleoperation.

## Hardware

### Bill of Materials

| Part | Quantity | Notes |
|------|----------|-------|
| Dynamixel XL330-M288-T | 2 | Yaw (ID 0) and Pitch (ID 1) motors |
| U2D2 | 1 | USB to Dynamixel adapter |
| XL330 Frame Set | 1 | Mechanical mounting |
| 3-pin Dynamixel cables | 2-3 | Daisy chain connection |
| USB-C cable | 1 | U2D2 to robot computer |

### Wiring

```
Robot USB Port
     │
   [U2D2]
     │
   ┌─┴─┐
   │   │
[Yaw] [Pitch]
 ID:0   ID:1
```

## Initial Motor Setup

### 1. Install Dynamixel Wizard 2.0

Download from [ROBOTIS](https://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_wizard2/)

```bash
# On your local PC (not robot - wizard may not work on Jetson due to glibc)
chmod +x DynamixelWizard2Setup_x64
./DynamixelWizard2Setup_x64
```

### 2. Configure Motors

Video tutorial: [YouTube](https://youtu.be/q3mCdYYJPNY?si=tGaAALMcXKZuwG5P)

1. Connect U2D2 to your PC
2. Grant serial port access:
   ```bash
   sudo chmod 666 /dev/ttyUSB0
   ```
3. Open Dynamixel Wizard and scan for motors
4. Configure each motor:
   - **Yaw motor**: Set ID to **0**
   - **Pitch motor**: Set ID to **1**
   - Both: Baud rate **57600** (or 2000000 for faster response)

## Robot Setup

### 1. Deploy Scripts to Robot

From your PC:

```bash
cd twist2_docker/twist2/robot_deploy
./deploy_to_robot.sh 192.168.123.164
```

This copies:
- `robot_peripherals.py` - Main control script (video + neck)
- `read_neck_position.py` - Position readout for calibration
- `run_peripherals.sh` - Wrapper with environment defaults
- `reset_zed_usb.sh` - USB reset for ZED camera issues

### 2. Serial Port Permissions

On the robot (required after each boot):

```bash
sudo chmod 666 /dev/ttyUSB0

# Or add permanent access:
sudo usermod -a -G dialout $USER
# Then logout and login again
```

## Calibration

### Finding the Center Position

1. Physically position the neck looking straight ahead
2. Read current encoder values:
   ```bash
   python3 read_neck_position.py
   # Press Enter to read positions
   ```
3. Note the values (e.g., YAW=1338, PITCH=695)

### Setting Defaults

Edit `run_peripherals.sh`:

```bash
YAW_CENTER=${YAW_CENTER:-1338}    # Your calibrated value
PITCH_CENTER=${PITCH_CENTER:-695}  # Your calibrated value
```

Or use environment variables:

```bash
YAW_CENTER=1338 PITCH_CENTER=695 ./run_peripherals.sh
```

## Usage

### Automatic (Recommended)

From PC, run:

```bash
# Terminal 1: Teleop (connects to PICO VR)
cd twist2
bash teleop_inspire.sh

# Terminal 2: RL policy + auto-starts peripherals on robot
cd twist2
bash sim2real_peripheral.sh
```

The `sim2real_peripheral.sh` script:
- SSHs to robot and starts `robot_peripherals.py`
- Runs the RL policy
- Automatically stops peripherals on Ctrl+C

### Manual

On the robot:

```bash
# With defaults
./run_peripherals.sh

# With custom settings
python3 robot_peripherals.py \
    --redis 192.168.123.222 \
    --yaw_center 1338 \
    --pitch_center 695 \
    --device /dev/ttyUSB0
```

## Command Line Options

`robot_peripherals.py` options:

| Option | Default | Description |
|--------|---------|-------------|
| `--redis` | 192.168.50.164 | PC IP running Redis/teleop |
| `--yaw_center` | 2048 | Yaw encoder center position |
| `--pitch_center` | 2048 | Pitch encoder center position |
| `--device` | /dev/ttyUSB0 | Dynamixel serial port |
| `--stream_port` | 12345 | Video streaming port |
| `--cmd_port` | 13579 | Command port for PICO |

## Troubleshooting

### Motor Not Moving

1. Check USB connection: `ls /dev/ttyUSB*`
2. Check permissions: `ls -la /dev/ttyUSB0`
3. Test with read script: `python3 read_neck_position.py`

### Motor Has Error (Red LED)

The script auto-reboots motors on startup. If issues persist:

```bash
# Read hardware error register
python3 read_neck_position.py
# Look for "hw_err" values
```

Common errors:
- `0x04`: Overload - reduce load or check for mechanical binding
- `0x08`: Overheating - let motor cool down

### Wrong Direction

- **Pitch inverted**: This is corrected in software (pitch_rad is negated)
- **Yaw wrong**: Check motor ID assignment (Yaw should be ID 0)

### Cannot Connect to Redis

1. Verify Redis is running on PC: `redis-cli ping`
2. Check Redis is bound to all interfaces (not just localhost)
3. Verify network connectivity: `ping <PC_IP>`

### ZED Camera Issues

If ZED shows blank or won't initialize after robot restart:

```bash
# Software USB reset
./reset_zed_usb.sh

# Or physically unplug/replug the USB cable
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         PICO VR                             │
│  XRobot App (displays video, sends head tracking)          │
└───────────────────────────┬─────────────────────────────────┘
                            │ WiFi
┌───────────────────────────▼─────────────────────────────────┐
│                     PC (Docker)                             │
│  ┌──────────────────────┐  ┌─────────────────────────────┐  │
│  │ xrobotoolkit-pc-svc  │  │ teleop_inspire.py           │  │
│  │ (video relay)        │  │ (head tracking → Redis)     │  │
│  └──────────────────────┘  └─────────────────────────────┘  │
│                                      │                      │
│                                      ▼ Redis                │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ sim2real (RL policy → robot body control)              │ │
│  └────────────────────────────────────────────────────────┘ │
└───────────────────────────┬─────────────────────────────────┘
                            │ Ethernet (192.168.123.x)
┌───────────────────────────▼─────────────────────────────────┐
│                    Robot (Jetson)                           │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ robot_peripherals.py                                   │ │
│  │ - ZED Mini → H.264 video → PICO                       │ │
│  │ - Redis (neck commands) → Dynamixel motors            │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Logs

Peripheral logs are saved to `~/logs/` on the robot:

```bash
# View recent logs
ssh unitree@192.168.123.164 'ls -lt ~/logs/ | head'

# Tail latest log
ssh unitree@192.168.123.164 'tail -f ~/logs/peripheral_*.log'
```

Log rotation keeps the last 100 log files automatically.
