#!/bin/bash
# Hybrid Teleop - Full Robot Control
# ===================================
# Complete teleop with hybrid locomotion, neck tracking, and Inspire hands.
#
# States:
#   idle        : Waiting for Pico VR data
#   preview     : MuJoCo preview (calibrate here)
#   teleop_full : Full body teleop - TWIST2 controls all joints
#   teleop_loco : Locomotion mode - legs walk via joystick, upper body tracks
#   paused      : Robot holds current pose
#
# Controls:
#   Right A        : preview → teleop → pause → teleop...
#   Left X         : Toggle teleop_full ↔ teleop_loco
#   Right A+B      : EMERGENCY SHUTDOWN
#   Left joystick  : Walk (in teleop_loco)
#   Right joystick : Rotate (in teleop_loco)
#   Triggers       : Close Inspire hands
#   Grips          : Open Inspire hands

# Make sure Redis is running
redis-cli ping > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Starting Redis server..."
    redis-server --daemonize yes
    sleep 1
fi

# Navigate to deployment directory
cd "$(dirname "$0")/deploy_real" || exit 1

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate gmr

# Configuration
actual_human_height=1.80
redis_ip="localhost"
target_fps=50

# Inspire hand IPs (on robot network)
inspire_left_ip="192.168.123.210"
inspire_right_ip="192.168.123.211"

echo ""
echo "============================================================"
echo "  HYBRID TELEOP - Full Robot Control"
echo "============================================================"
echo ""
echo "Complete teleop: body + neck + locomotion + Inspire hands"
echo ""
echo "Prerequisites:"
echo "  1. XRobotToolkit app running on Pico"
echo "  2. Pico connected to this PC"
echo "  3. RoboMimic_Deploy mounted at /workspace/RoboMimic_Deploy"
echo "  4. For robot: run sim2real_full.sh in another terminal"
echo ""
echo "Controls:"
echo "  Right A        : preview → teleop → pause → teleop..."
echo "  Left X         : Toggle teleop_full ↔ teleop_loco"
echo "  Right A+B      : EMERGENCY SHUTDOWN"
echo "  Left joystick  : Walk (in teleop_loco)"
echo "  Right joystick : Rotate (in teleop_loco)"
echo "  Triggers       : Close hands"
echo "  Grips          : Open hands"
echo ""
echo "Workflow:"
echo "  1. Run this script (PC side)"
echo "  2. In another terminal: sim2real_full.sh (robot side)"
echo "  3. Calibrate in preview mode"
echo "  4. Press Right A → teleop"
echo "  5. Press Left X → walking mode"
echo "  6. Use triggers/grips for hands"
echo "============================================================"
echo ""

# Check if RoboMimic_Deploy exists
if [ ! -d "/workspace/RoboMimic_Deploy" ]; then
    echo "ERROR: RoboMimic_Deploy not mounted in container!"
    echo ""
    echo "Update docker-compose.yml and restart container:"
    echo "  docker-compose down && docker-compose up -d"
    exit 1
fi

# Run the hybrid teleop with Inspire hands
python teleop_hybrid.py \
    --robot unitree_g1 \
    --actual_human_height $actual_human_height \
    --redis_ip $redis_ip \
    --target_fps $target_fps \
    --smooth --smooth_window_size 4 \
    --use_inspire_hands \
    --inspire_left_ip $inspire_left_ip \
    --inspire_right_ip $inspire_right_ip

echo ""
echo "Hybrid teleop stopped."

