#!/bin/bash
# Launch script for Merged Teleop with dex-retargeting
# Proper IK-based finger retargeting (no calibration needed!)

set -e

echo "============================================================"
echo "  MERGED TELEOP (dex-retargeting)"
echo "  IK-based finger tracking - NO CALIBRATION NEEDED!"
echo "============================================================"
echo ""
echo "Prerequisites:"
echo "  1. XRobotToolkit app running on Pico"
echo "  2. Pico connected to this PC"
echo "  3. For robot: run sim2real_full.sh in another terminal"
echo ""
echo "Unitree Controller Mapping:"
echo "  A       : Cycle states (idle → preview → teleop → pause)"
echo "  B       : Toggle WALK ↔ BALANCE mode"
echo "  X       : Toggle UPPER BODY (arms+hands) freeze"
echo "  Select  : EMERGENCY SHUTDOWN"
echo ""
echo "Pico VR:"
echo "  Body    : Full body motion tracking"
echo "  Hands   : Finger tracking → Inspire hands (dex-retargeting IK)"
echo ""
echo "============================================================"
echo ""

# Navigate to deployment directory
cd "$(dirname "$0")/deploy_real" || exit 1

# Activate conda environment (gmr has Python 3.10 + dex-retargeting)
eval "$(conda shell.bash hook)"
conda activate gmr

# Make sure Redis is running
redis-cli ping > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Starting Redis server..."
    redis-server --daemonize yes
    sleep 1
fi

# Check if RoboMimic_Deploy exists
if [ ! -d "/workspace/RoboMimic_Deploy" ]; then
    echo "ERROR: RoboMimic_Deploy not mounted in container!"
    echo ""
    echo "Update docker-compose.yml and restart container:"
    echo "  docker-compose down && docker-compose up -d"
    exit 1
fi

# Check if dex-retargeting is installed
python -c "from dex_retargeting.constants import RobotName" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing dex-retargeting..."
    pip install dex-retargeting
fi

# Check if Inspire hand URDF exists
if [ ! -f "inspire_hand/inspire_hand_right.urdf" ]; then
    echo "ERROR: Inspire hand URDF not found!"
    echo "Copy from XRoboToolkit-Teleop-Sample-Python/assets/inspire_hand/"
    exit 1
fi

# Configuration
actual_human_height=1.80
redis_ip="localhost"
target_fps=50

# Inspire hand IPs (on robot network)
inspire_left_ip="192.168.123.210"
inspire_right_ip="192.168.123.211"

# Run the merged teleop with dex-retargeting
python teleop_merged_dex.py \
    --robot unitree_g1 \
    --actual_human_height $actual_human_height \
    --redis_ip $redis_ip \
    --target_fps $target_fps \
    --smooth --smooth_window_size 4 \
    --use_inspire_hands \
    --inspire_left_ip $inspire_left_ip \
    --inspire_right_ip $inspire_right_ip \
    "$@"

echo ""
echo "Teleop ended."

