#!/bin/bash
# Sim2Sim V6.5 - Default architecture + Jerk penalty training
#
# V6.5 Key features:
# - Default student architecture (1397 dims = 127 × 11)
# - NO separate future observations (unlike V6.3)
# - Trained with jerk penalty for smoother actions
#
# Usage: bash sim2sim_v6_5.sh [policy_path]

SCRIPT_DIR=$(dirname $(realpath $0))

# Default to V6.5 trained model
ckpt_path=${1:-${SCRIPT_DIR}/legged_gym/logs/h1/student_v6_5_wandb/model_17500.onnx}

if [ ! -f "$ckpt_path" ]; then
    echo "ERROR: V6.5 policy not found: $ckpt_path"
    echo "Train V6.5 first or provide a valid path"
    exit 1
fi

echo "=============================================="
echo "  Sim2Sim V6.5 (Default + Jerk Penalty)"
echo "=============================================="
echo "  Policy: ${ckpt_path}"
echo ""
echo "  ARCHITECTURE:"
echo "    - Obs: 1397 dims (127 × 11)"
echo "    - Current + 10 history frames"
echo "    - NO separate future obs"
echo ""
echo "  TRAINING:"
echo "    - Default student architecture"
echo "    - Jerk penalty: -0.1 (smoother actions)"
echo ""
echo "  Make sure motion server is running!"
echo "=============================================="

cd deploy_real

python server_low_level_g1_sim_v6_5.py \
    --xml ../assets/g1/g1_sim2sim_29dof.xml \
    --policy ${ckpt_path} \
    --device cuda \
    --redis_ip localhost \
    --robot unitree_g1_with_hands


