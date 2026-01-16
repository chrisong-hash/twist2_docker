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

# ===================================
# Complete teleop with hybrid locomotion, neck tracking, and Inspire hands.
# Uses GROOT GearWBC for stable walking with arbitrary arm poses.
#
# States:
#   idle    : Waiting for Pico VR data
#   preview : MuJoCo preview (calibrate here)
#   teleop  : Active teleoperation
#
# Controls (Pico VR Controller):
#   Right A (release)   : Cycle idle → preview → teleop → pause → teleop...
#   Right A + Left X    : Toggle upper body freeze (arms+hands)
#   Right A + Left Y    : Toggle walk ↔ balance mode
#   Right A+B           : EMERGENCY SHUTDOWN
#   Left Trigger        : Open LEFT hand
#   Right Trigger       : Open RIGHT hand
#   Left Grip           : Close LEFT hand (thumb lags 1 sec)
#   Right Grip          : Close RIGHT hand (thumb lags 1 sec)
#   Left X + L Trigger  : LEFT thumb outward
#   Left X + R Trigger  : RIGHT thumb outward
#   Left X + L Grip     : LEFT thumb inward
#   Left X + R Grip     : RIGHT thumb inward
#   Left joystick       : Walk (in walk mode)
#   Right joystick      : Rotate (in walk mode)

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
echo "Prerequisites:"
echo "  1. XRobotToolkit app running on Pico"
echo "  2. Pico connected to this PC"
echo "  3. RoboMimic_Deploy mounted at /workspace/RoboMimic_Deploy"
echo "  4. For robot: run sim2real_full.sh in another terminal"
echo ""
echo "Controls (Pico VR - NOT Unitree remote!):"
echo "  Right A (release) : Cycle idle → preview → teleop → pause..."
echo "  Right A + Left X  : Toggle upper body freeze (arms+hands)"
echo "  Right A + Left Y  : Toggle walk ↔ balance mode"
echo "  Right A+B         : EMERGENCY SHUTDOWN"
echo "  Left Trigger      : Open LEFT hand"
echo "  Right Trigger     : Open RIGHT hand"
echo "  Left Grip         : Close LEFT hand (thumb lags 1 sec)"
echo "  Right Grip        : Close RIGHT hand (thumb lags 1 sec)"
echo "  Left X + L Trigger: LEFT thumb outward"
echo "  Left X + R Trigger: RIGHT thumb outward"
echo "  Left X + L Grip   : LEFT thumb inward"
echo "  Left X + R Grip   : RIGHT thumb inward"
echo "  Left joystick     : Walk (in walk mode)"
echo "  Right joystick    : Rotate (in walk mode)"
echo ""
echo "Workflow:"
echo "  1. Run this script (PC side - teleop controller)"
echo "  2. In another terminal: sim2real_full.sh (robot side)"
echo "  3. Pico connects → auto enters PREVIEW (MuJoCo shows your motion)"
echo "  4. Press Right A on Pico → TELEOP (robot follows you)"
echo "  5. Press Right A + Left Y → WALK mode (joystick locomotion)"
echo "  6. Press Right A again → PAUSE (robot freezes)"
echo "  7. Reposition your body, press Right A → UNPAUSE (recalibrates)"
echo "============================================================"
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
