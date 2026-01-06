#!/bin/bash
# Hybrid sim2real - uses LocoMode leg positions directly, TWIST2 policy for upper body
#
# Use this with hybrid_teleop.sh to enable joystick walking!

source ~/miniconda3/bin/activate twist2

SCRIPT_DIR=$(dirname $(realpath $0))
ckpt_path=${SCRIPT_DIR}/assets/ckpts/twist2_1017_20k.onnx

# change the network interface name to your own that connects to the robot
net=enp4s0

cd deploy_real

echo ""
echo "============================================================"
echo "  HYBRID SIM2REAL (REAL ROBOT)"
echo "============================================================"
echo ""
echo "Mode: Leg positions from LocoMode (joystick control)"
echo "       Upper body from TWIST2 policy"
echo ""
echo "Use with: hybrid_teleop.sh"
echo ""
echo "SAFETY: Press B button on Pico to EMERGENCY SHUTDOWN both"
echo "        teleop and this robot server!"
echo "============================================================"
echo ""

# Clear stale Redis keys from previous sessions to prevent unwanted LocoMode activation
echo "Clearing stale Redis keys..."
redis-cli DEL teleop_state_info loco_vel_cmd > /dev/null 2>&1
echo "Redis keys cleared. Robot will use TWIST2 until hybrid_teleop.sh sets teleop_loco state."
echo ""

python server_low_level_g1_real.py \
    --policy ${ckpt_path} \
    --net ${net} \
    --device cuda \
    --use_hand \
    --hybrid_loco_mode \
    --smooth_body 0.5



