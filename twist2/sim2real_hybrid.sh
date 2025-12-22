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
echo "  HYBRID SIM2REAL"
echo "============================================================"
echo ""
echo "Mode: Leg positions from mimic_obs (LocoMode)"
echo "       Upper body from TWIST2 policy"
echo ""
echo "Use with: hybrid_teleop.sh"
echo "============================================================"
echo ""

python server_low_level_g1_real.py \
    --policy ${ckpt_path} \
    --net ${net} \
    --device cuda \
    --use_hand \
    --hybrid_loco_mode


