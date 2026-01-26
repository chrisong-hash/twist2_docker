#!/bin/bash
# Hybrid Teleop - Full Robot Control
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

# Velocity scaling for GROOT GearWBC (adjust to taste)
# Lower values = slower walking, higher = faster
# After cmd_scale [2.0, 2.0, 0.5]: actual speed = scale * 2.0 m/s (or * 0.5 for yaw)
vel_scale_forward=0.3    # 0.3 → max 0.6 m/s forward
vel_scale_backward=0.1   # 0.1 → max 0.2 m/s backward (very conservative for stability)
vel_scale_strafe=0.25    # 0.25 → max 0.5 m/s strafe
vel_scale_yaw=0.4        # 0.4 → max 0.2 rad/s yaw

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

# Check if RoboMimic_Deploy exists
if [ ! -d "/workspace/RoboMimic_Deploy" ]; then
    echo "ERROR: RoboMimic_Deploy not mounted in container!"
    echo ""
    echo "Update docker-compose.yml and restart container:"
    echo "  docker-compose down && docker-compose up -d"
    exit 1
fi

# Run the hybrid teleop (Inspire hands enabled by default)
python teleop_hybrid.py \
    --robot unitree_g1 \
    --actual_human_height $actual_human_height \
    --redis_ip $redis_ip \
    --target_fps $target_fps \
    --smooth --smooth_window_size 4 \
    --use_inspire_hands \
    --inspire_left_ip $inspire_left_ip \
    --inspire_right_ip $inspire_right_ip \
    --vel_scale_forward $vel_scale_forward \
    --vel_scale_backward $vel_scale_backward \
    --vel_scale_strafe $vel_scale_strafe \
    --vel_scale_yaw $vel_scale_yaw

echo ""
echo "Hybrid teleop stopped."

