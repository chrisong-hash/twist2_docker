#!/bin/bash
# Hybrid Locomotion + Teleoperation
# ==================================
# Same workflow as teleop_inspire.sh but with joystick walking!
#
# Workflow:
# 1. Start XRobotToolkit app on Pico
# 2. Connect Pico to this PC via the app
# 3. Run this script
# 4. MuJoCo preview shows your motion - calibrate until tracking works
# 5. Press Right A button to enter teleop mode
# 6. HOLD Right Trigger + use joystick to walk
# 7. Then run sim2real.sh in another terminal to enable robot

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
actual_human_height=${HUMAN_HEIGHT:-1.65}
redis_ip="localhost"
target_fps=50

echo ""
echo "============================================================"
echo "  HYBRID LOCOMOTION + TELEOPERATION"
echo "============================================================"
echo ""
echo "This script follows the same workflow as teleop_inspire.sh"
echo "but adds joystick-based walking via RoboMimic LocoMode!"
echo ""
echo "Prerequisites:"
echo "  1. XRobotToolkit app running on Pico"
echo "  2. Pico connected to this PC"
echo "  3. RoboMimic_Deploy cloned at /home/robo/CodeSpace/RoboMimic_Deploy"
echo ""
echo "Controls:"
echo "  Right A button  : Toggle preview/teleop mode"
echo "  Left A button   : Exit"
echo "  Left joystick   : Walk (forward/back/strafe)"
echo "  Right joystick  : Rotate"
echo "  Right trigger   : HOLD to enable leg locomotion"
echo ""
echo "Workflow:"
echo "  1. Move until MuJoCo reflects your motion (calibration)"
echo "  2. Press Right A → enters teleop mode"
echo "  3. In another terminal: cd /workspace/twist2 && bash sim2real.sh"
echo "  4. HOLD Right Trigger + joystick to walk!"
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
    --target_fps $target_fps

echo ""
echo "Hybrid teleop stopped."
