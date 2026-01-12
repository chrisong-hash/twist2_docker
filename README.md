# TWIST2 Docker Environment

A Docker environment for TWIST2 humanoid robot control and teleoperation with Unitree G1.

## Quick Start

```bash
# Clone with submodules
git clone --recursive <your-repo>
cd twist2_docker

# Download IsaacGym (requires NVIDIA registration)
# https://developer.nvidia.com/isaac-gym
# Extract to: twist2_docker/isaacgym/

# Build and run
docker compose build
docker compose up -d
docker exec -it twist2 bash
```

## Prerequisites

- Docker with NVIDIA GPU support
- NVIDIA drivers installed on host
- X11 display for GUI
- For real robot: Unitree G1 hardware

## Setup

### 1. Allow X11 Access

```bash
xhost +local:docker
```

### 2. Build Container

```bash
docker compose build
```

### 3. Start Container

```bash
docker compose up -d
```

### 4. Enter Container

```bash
docker exec -it twist2 bash
```

## Usage

### Teleoperation with Real Robot

**Requirements:**
- PICO VR controller with XRobotStreamer
- Unitree G1 robot connected via Ethernet
- Network configured: IP `192.168.123.222`, netmask `255.255.255.0`
- Robot in dev mode (L2+R2 on remote)

**Terminal 1 - Low-level controller:**
```bash
docker exec -it twist2 bash
cd /workspace/twist2
bash sim2real.sh
```

**Terminal 2 - Teleoperation:**
```bash
docker exec -it twist2 bash
cd /workspace/twist2
bash teleop.sh

# Or with Inspire hands (trigger-based control):
bash teleop_inspire.sh
```

### Simulation Testing (Sim2Sim)

**Terminal 1 - Low-level controller:**
```bash
docker exec -it twist2 bash
cd /workspace/twist2
bash sim2sim.sh
```

**Terminal 2 - Motion server:**
```bash
docker exec -it twist2 bash
cd /workspace/twist2
bash run_motion_server.sh
```

### Training

```bash
docker exec -it twist2 bash
cd /workspace/twist2

# Quick test
bash train.sh test_docker cuda:0

# Full training (requires dataset)
bash train.sh my_experiment cuda:0
```

## Configuration

### Increasing Robot Gains

To increase power for walking or other movements, edit `twist2/deploy_real/robot_control/configs/g1.yaml`:

```yaml
kps: [150, 150, 150, 200, 60, 60,  # left leg (increase for more power)
      150, 150, 150, 200, 60, 60,  # right leg
      # ... other joints
     ]

action_scale: 0.75  # increase for more aggressive movements
```

## Neck Control (Dynamixel)

The robot uses two Dynamixel XL330-M288-T motors for neck control (yaw and pitch), connected via U2D2 adapter.

### Hardware Setup

1. Connect U2D2 to robot's USB port
2. Motor IDs: **Yaw = 0**, **Pitch = 1**
3. Baud rate: **57600** (or 2Mbps if configured)

### Deploy to Robot

From the PC, copy the peripheral scripts to the robot:

```bash
cd twist2/robot_deploy
./deploy_to_robot.sh 192.168.123.164
```

### Manual Neck Control on Robot

```bash
# SSH to robot
ssh unitree@192.168.123.164

# Grant serial port access (required after each boot)
sudo chmod 666 /dev/ttyUSB0

# Read current motor positions (for calibration)
python3 read_neck_position.py

# Run peripherals manually (video + neck)
python3 robot_peripherals.py --redis 192.168.123.222

# Or use the wrapper script
./run_peripherals.sh
```

### Calibration

To find your neck center position:

1. Manually position the neck to look straight ahead
2. Run `python3 read_neck_position.py` on the robot
3. Note the YAW and PITCH values
4. Update the defaults in `run_peripherals.sh` or use env vars:

```bash
YAW_CENTER=1338 PITCH_CENTER=695 ./run_peripherals.sh
```

### Automatic Startup (via sim2real_peripheral.sh)

The `sim2real_peripheral.sh` script automatically:
- SSHs to the robot and starts `robot_peripherals.py`
- Runs the RL policy on PC
- Cleans up peripherals when you Ctrl+C

```bash
cd twist2
bash sim2real_peripheral.sh
```

### Troubleshooting

- **Motor not responding**: Check `/dev/ttyUSB0` exists and has write permission
- **Wrong direction**: Pitch is inverted in software; yaw follows head direction
- **Motor error (red LED)**: Script auto-reboots motors on startup

**Full documentation:** See [twist2/doc/TWIST2_NECK.md](twist2/doc/TWIST2_NECK.md)

## Directory Structure

```
twist2_docker/
├── README.md
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── isaacgym/              # Isaac Gym installation
├── GMR/                   # General Motion Retargeting
├── unitree_sdk2/          # Unitree SDK
└── twist2/                # TWIST2 source code
    ├── assets/
    │   └── ckpts/         # Pretrained checkpoints
    ├── deploy_real/       # Real robot deployment (PC-side)
    │   ├── robot_control/ # Robot control modules
    │   └── xrobot_teleop_inspire.py
    ├── robot_deploy/      # Robot deployment (robot-side)
    │   ├── robot_peripherals.py  # ZED video + neck control
    │   ├── read_neck_position.py # Calibration helper
    │   ├── run_peripherals.sh    # Startup wrapper
    │   ├── reset_zed_usb.sh      # USB reset for ZED issues
    │   └── deploy_to_robot.sh    # Copy files to robot
    ├── legged_gym/        # Training environments
    ├── rsl_rl/            # RL algorithms
    ├── teleop_inspire.sh  # Teleoperation with Inspire hands
    ├── sim2real_peripheral.sh # RL + auto-peripheral management
    ├── sim2real.sh        # RL policy only (legacy)
    ├── sim2sim.sh         # Simulation test
    └── train.sh           # Training script
```

## Common Commands

```bash
# Start container
docker compose up -d

# Stop container
docker compose down

# Rebuild after changes
docker compose build --no-cache
docker compose up -d --force-recreate

# View logs
docker logs twist2

# Check Redis
docker exec twist2 redis-cli ping

# Check GPU
docker exec twist2 nvidia-smi
```

## Troubleshooting

### Cannot open display
```bash
xhost +local:docker
```

### CUDA not available
```bash
# Test NVIDIA Docker
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi
```

### Redis connection refused
```bash
docker exec twist2 redis-server --daemonize yes
```

### Robot not moving
- Check gains in `g1.yaml`
- Verify network connection to robot
- Ensure robot is in dev mode
- Check Redis connection

## License

MIT License - See LICENSE file for details

## Acknowledgments

- TWIST2 repository: [amazon-far/TWIST2](https://github.com/amazon-far/TWIST2)
- Isaac Gym by NVIDIA
- Unitree Robotics

---

**Questions?** yanjieze@stanford.edu (original TWIST2 author)
