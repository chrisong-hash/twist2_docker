#!/bin/bash
# Hybrid sim2sim - uses LocoMode leg positions directly, TWIST2 policy for upper body
#
# Use this with hybrid_teleop.sh to enable joystick walking in simulation!

SCRIPT_DIR=$(dirname $(realpath $0))
ckpt_path=${SCRIPT_DIR}/assets/ckpts/twist2_1017_20k.onnx

cd deploy_real

echo ""
echo "============================================================"
echo "  HYBRID SIM2SIM"
echo "============================================================"
echo ""
echo "Mode: Leg positions from LocoMode (joystick control)"
echo "       Upper body from TWIST2 policy"
echo ""
echo "Use with: hybrid_teleop.sh"
echo "============================================================"
echo ""

python server_low_level_g1_sim.py \
    --xml ../assets/g1/g1_sim2sim_29dof.xml \
    --policy ${ckpt_path} \
    --device cuda \
    --measure_fps 1 \
    --policy_frequency 100 \
    --limit_fps 1 \
    --hybrid_loco_mode

