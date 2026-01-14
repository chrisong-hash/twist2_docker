#!/bin/bash
# sim2real with Holosoma natural gait policy (REAL ROBOT)
#
# This uses the Holosoma V3 policy for upright, natural walking
# instead of RoboMimic's crouched walking.
#
# Usage:
#   bash sim2real_holosoma.sh
#
# Controls:
#   - Run hybrid_teleop.sh to control via Pico joystick
#   - Right Grip: Toggle walking mode
#   - Left A: Toggle pause
#   - B: EMERGENCY SHUTDOWN
#
# Key difference from RoboMimic:
#   - Knees stay at 8-15° (natural) vs 40° (crouched)
#   - More human-like walking gait

source ~/miniconda3/bin/activate twist2

SCRIPT_DIR=$(dirname $(realpath $0))

# TWIST2 upper body policy
twist2_policy=${SCRIPT_DIR}/assets/ckpts/twist2_1017_20k.onnx

# Holosoma natural gait policy for legs
holosoma_policy=${SCRIPT_DIR}/assets/ckpts/holosoma_natural_gait_v3.onnx

# Network interface - change to match your setup
net=enp4s0

# Check if Holosoma model exists
if [ ! -f "$holosoma_policy" ]; then
    echo "ERROR: Holosoma model not found at $holosoma_policy"
    echo ""
    echo "Please copy the model first:"
    echo "  cp /home/robo/CodeSpace/natural_gait_g1/trained_models/v3_stable/model_0050000.onnx \\"
    echo "     ${holosoma_policy}"
    exit 1
fi

cd deploy_real

echo ""
echo "============================================================"
echo "  HOLOSOMA NATURAL GAIT - REAL ROBOT"
echo "============================================================"
echo ""
echo "Upper body: TWIST2 (${twist2_policy})"
echo "Legs:       Holosoma V3 Natural Gait (${holosoma_policy})"
echo ""
echo "Use with: hybrid_teleop.sh"
echo ""
echo "SAFETY: Press B button on Pico to EMERGENCY SHUTDOWN both"
echo "        teleop and this robot server!"
echo "============================================================"
echo ""

# Clear stale Redis keys from previous sessions
echo "Clearing stale Redis keys..."
redis-cli DEL teleop_state_info loco_vel_cmd robot_shutdown > /dev/null 2>&1
echo "Redis keys cleared. Robot will use TWIST2 until hybrid_teleop.sh sets teleop_loco state."
echo ""

python server_low_level_g1_real.py \
    --policy ${twist2_policy} \
    --net ${net} \
    --device cuda \
    --use_hand \
    --hybrid_loco_mode \
    --use_holosoma \
    --holosoma_model ${holosoma_policy} \
    --smooth_body 0.5


