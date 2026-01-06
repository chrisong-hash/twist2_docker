#!/bin/bash
# Hybrid Locomotion + Teleoperation v2
# =====================================
# Full body teleop with optional joystick walking mode.
#
# States:
#   idle        : Waiting for Pico VR data
#   preview     : MuJoCo preview (calibrate here)
#   teleop_full : Full body teleop (default) - TWIST2 controls all joints
#   teleop_loco : Locomotion mode - legs walk via joystick, upper body tracks
#   paused      : Robot at default standing pose
#
# Workflow:
# 1. Start XRobotToolkit app on Pico
# 2. Connect Pico to this PC via the app
# 3. Run this script
# 4. MuJoCo preview shows your motion - calibrate until tracking works
# 5. Press Right A → enters teleop_full (full body teleop)
# 6. Press Right Grip → switches to teleop_loco (walking mode)
# 7. Use joysticks to walk in teleop_loco
# 8. Press Right Grip again → back to teleop_full
# 9. Press Left A anytime → pause/unpause
#
# All transitions have smooth 1-second interpolation!

# Make sure Redis is running
redis-cli ping > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Starting Redis server..."
    redis-server --daemonize yes
    sleep 1
fi

# Navigate to deployment directory
cd "$(dirname "$0")/deploy_real" || exit 1

# Activate conda environment (same as teleop_inspire.sh)
eval "$(conda shell.bash hook)"
conda activate gmr

# Configuration
actual_human_height=1.80
redis_ip="localhost"
target_fps=50

echo ""
echo "============================================================"
echo "  HYBRID LOCOMOTION + TELEOPERATION v2"
echo "============================================================"
echo ""
echo "Full body teleop with optional joystick walking mode."
echo "All mode transitions have smooth 1-second interpolation!"
echo ""
echo "Prerequisites:"
echo "  1. XRobotToolkit app running on Pico"
echo "  2. Pico connected to this PC"
echo "  3. RoboMimic_Deploy mounted at /workspace/RoboMimic_Deploy"
echo ""
echo "Controls:"
echo "  Right A (key_one)  : Toggle preview ↔ teleop"
echo "  Right Grip         : Toggle teleop_full ↔ teleop_loco"
echo "  Left A (key_one)   : Toggle pause"
echo "  B button (key_two) : EMERGENCY SHUTDOWN (stops teleop + robot server)"
echo "  Left joystick      : Walk (only in teleop_loco)"
echo "  Right joystick     : Rotate (only in teleop_loco)"
echo ""
echo "States:"
echo "  teleop_full : Full body teleop (default)"
echo "  teleop_loco : Walking mode (joystick controls legs)"
echo "  paused      : Robot at standing pose"
echo ""
echo "Workflow:"
echo "  1. Calibrate until MuJoCo reflects your motion"
echo "  2. Press Right A → enters teleop_full"
echo "  3. In another terminal: cd /workspace/twist2 && bash sim2real_hybrid.sh"
echo "  4. Press Right Grip → teleop_loco (walking mode)"
echo "  5. Use joysticks to walk"
echo "  6. Press Right Grip → back to teleop_full"
echo "  7. Press Left A → pause/unpause"
echo "============================================================"
echo ""

# Check if RoboMimic_Deploy exists (mounted at /workspace/RoboMimic_Deploy)
if [ ! -d "/workspace/RoboMimic_Deploy" ]; then
    echo "ERROR: RoboMimic_Deploy not mounted in container!"
    echo ""
    echo "You need to restart the Docker container after updating docker-compose.yml:"
    echo "  cd ~/CodeSpace/twist2_docker"
    echo "  docker-compose down"
    echo "  docker-compose up -d"
    echo "  docker exec -it twist2 bash"
    exit 1
fi

# Run the hybrid teleop
python hybrid_loco_teleop.py \
    --robot unitree_g1 \
    --actual_human_height $actual_human_height \
    --redis_ip $redis_ip \
    --target_fps $target_fps \
    --smooth --smooth_window_size 4

echo ""
echo "Hybrid teleop stopped."
