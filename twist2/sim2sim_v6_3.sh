#!/bin/bash
# Sim2Sim V6.3 - Student with Jerk Penalty + Wrist Dampening When Static
#
# V6.3 inherits from V6.2 (future observations) + jerk penalty reward
# Trained for 27.5k iterations on RunPod
#
# VALIDATED CONFIG:
#   - Baseline PD gains (stable)
#   - Wrist-only dampening when static (reduces jitter without affecting balance)
#   - Static: threshold=0.01, blend=0.9, scale=0.1
#
# Test result: Waist Z min=0.780 (stable with wrist dampening)
#
# Requires:
#   1. Motion Server (run_motion_server_raw.sh) publishing actual future observations
#   2. V6.3 student ONNX policy
#
# Usage: bash sim2sim_v6_3.sh [policy_path]

SCRIPT_DIR=$(dirname $(realpath $0))

# Default to V6.3 student policy
ckpt_path=${1:-/workspace/twist2/legged_gym/logs/g1_priv_mimic/v6_3_fixed/model_27500.onnx}

echo "=============================================="
echo "  Sim2Sim V6.3 (Wrist Dampening When Static)"
echo "=============================================="
echo "  Policy: ${ckpt_path}"
echo ""
echo "  TRAINING:"
echo "    - Future observations (0.1s, 0.3s, 0.5s ahead)"
echo "    - Jerk penalty (action_jerk = -0.1)"
echo ""
echo "  DEPLOYMENT:"
echo "    - Wrist dampening when static (blend=0.9, scale=0.1)"
echo "    - Waist Z: min=0.780 (stable)"
echo ""
echo "  Make sure Raw Motion Server is running!"
echo "  (bash run_motion_server_raw.sh)"
echo "=============================================="

cd deploy_real

python server_low_level_g1_sim_v6_3.py \
    --xml ../assets/g1/g1_sim2sim_29dof.xml \
    --policy ${ckpt_path} \
    --device cuda \
    --redis_ip localhost \
    --robot unitree_g1_with_hands

