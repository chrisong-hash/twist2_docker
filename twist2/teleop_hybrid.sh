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
# Controls:
#   Left X (tap)        : Toggle preview ↔ teleop
#   Right A (teleop)    : Toggle upper body (tracking ↔ frozen)
#   Right B (teleop)    : Toggle lower body (standing ↔ walking)
#   Right A+B (hold 1s) : EMERGENCY SHUTDOWN
#   Left joystick       : Walk (in walking mode)
#   Right joystick      : Rotate (in walking mode)
#   Left trigger        : Toggle left hand (open ↔ closed)
#   Right trigger       : Toggle right hand (open ↔ closed)

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
echo "  Left X         : Toggle preview ↔ teleop"
echo "  Right A        : Toggle upper body (tracking ↔ frozen)"
echo "  Right B        : Toggle lower body (standing ↔ walking)"
echo "  Right A+B      : EMERGENCY SHUTDOWN (hold 1s)"
echo "  Left joystick  : Walk (in walking mode)"
echo "  Right joystick : Rotate (in walking mode)"
echo "  Left trigger   : Toggle left hand (open ↔ closed)"
echo "  Right trigger  : Toggle right hand (open ↔ closed)"
echo ""
echo "Workflow:"
echo "  1. Run this script (PC side)"
echo "  2. In another terminal: sim2real_full.sh (robot side)"
echo "  3. Pico connects → auto enters preview"
echo "  4. Press Left X → teleop (robot follows you)"
echo "  5. Press Right B → walking mode (joystick locomotion)"
echo "  6. Press Left X → preview (robot freezes)"
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
    --inspire_right_ip $inspire_right_ip

echo ""
echo "Hybrid teleop stopped."

