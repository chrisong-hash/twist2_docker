#!/bin/bash

# TWIST2 Hybrid Teleoperation Launch Script
# Combines PICO controller trigger-based control with Manus glove fine control

echo "=============================================="
echo "  TWIST2 Hybrid Teleop (PICO + Manus Gloves)"
echo "=============================================="

# Check XRoboToolkit PC Service (informational only)
if pgrep -f -i "xrobo" > /dev/null 2>&1; then
    echo "[✓] XRoboToolkit PC Service detected"
else
    echo "[!] XRoboToolkit PC Service may not be running"
    echo "    If PICO connection fails, start it from Applications menu"
    echo ""
fi

# Make sure Redis is running
redis-cli ping > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Starting Redis server..."
    redis-server --daemonize yes
    sleep 1
fi
echo "[✓] Redis ready"
echo ""

# Navigate to deployment directory
cd "$(dirname "$0")/deploy_real" || exit 1

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate gmr

# Configuration
actual_human_height=1.80
redis_ip="localhost"

# Inspire hand IPs
inspire_left_ip="192.168.123.210"
inspire_right_ip="192.168.123.211"

# Xsens UDP port for Manus gloves
xsens_port=9763

echo "Configuration:"
echo "  Robot: unitree_g1"
echo "  Human height: ${actual_human_height}m"
echo "  Redis: ${redis_ip}"
echo "  Inspire hands: L=${inspire_left_ip}, R=${inspire_right_ip}"
echo "  Xsens port: ${xsens_port}"
echo ""
echo "Boot Sequence:"
echo "  1. Press Unitree START button → Ready state"
echo "  2. Press Unitree A button → Preop mode"
echo ""
echo "Teleop Modes:"
echo "  - PICO Right A: Enter PICO finger mode (trigger-based)"
echo "  - Unitree A: Enter Manus glove mode (fine control)"
echo "  - PICO Right A: Return to preop (from either mode)"
echo ""
echo "=============================================="
echo ""

# Run hybrid teleoperation
python xrobot_teleop_hybrid.py \
    --robot unitree_g1 \
    --actual_human_height $actual_human_height \
    --redis_ip $redis_ip \
    --target_fps 100 \
    --use_inspire_hands \
    --inspire_left_ip $inspire_left_ip \
    --inspire_right_ip $inspire_right_ip \
    --enable_manus \
    --xsens_port $xsens_port

echo ""
echo "Hybrid teleoperation stopped."
