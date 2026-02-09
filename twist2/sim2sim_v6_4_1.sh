#!/bin/bash
# Sim2Sim V6.4.1 - Privileged Predictor Student (NO future observations)
#
# V6.4.1 learns to predict privileged info from history
# NO motion server needed for standing still!
#
# Observation: 1397 dims (127*11 = current + history)
# NO future observations - predictor estimates internally
#
# Usage: bash sim2sim_v6_4_1.sh [policy_path]

SCRIPT_DIR=$(dirname $(realpath $0))

ckpt_path=${1:-/workspace/twist2/legged_gym/logs/h1/v6_4_1_priv_pred/model_15000.onnx}

echo "=============================================="
echo "  Sim2Sim V6.4.1 (Privileged Predictor)"
echo "=============================================="
echo "  Policy: ${ckpt_path}"
echo ""
echo "  Architecture:"
echo "    - Input: 1397 dims (127×11)"
echo "    - NO future observations"
echo "    - Predictor estimates privileged info from history"
echo ""
echo "  NO motion server needed for standing test!"
echo "=============================================="

cd deploy_real

python server_low_level_g1_sim_v6_4_1.py \
    --xml ../assets/g1/g1_sim2sim_29dof.xml \
    --policy ${ckpt_path} \
    --device cuda \
    --redis_ip localhost \
    --robot unitree_g1_with_hands


