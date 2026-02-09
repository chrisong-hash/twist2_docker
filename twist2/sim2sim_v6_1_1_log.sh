#!/bin/bash
# Sim2Sim V6.1.1 with LOGGING
#
# Records all input/output data for analysis.
# Runs for 500 steps (10 seconds at 50Hz) then saves logs.
#
# Usage: bash sim2sim_v6_1_1_log.sh [policy_path] [max_steps]

SCRIPT_DIR=$(dirname $(realpath $0))

ckpt_path=${1:-${SCRIPT_DIR}/model_17500.onnx}
max_steps=${2:-500}

echo "=============================================="
echo "  Sim2Sim V6.1.1 with LOGGING"
echo "=============================================="
echo "  Policy: ${ckpt_path}"
echo "  Max steps: ${max_steps}"
echo "  Logs saved to: ./logs_v6_1_1/"
echo ""
echo "  NOTE: Using last Redis frame (static pose test)"
echo "=============================================="

cd deploy_real

python server_low_level_g1_sim_v6_1_1_log.py \
    --xml ../assets/g1/g1_sim2sim_29dof.xml \
    --policy ${ckpt_path} \
    --device cuda \
    --redis_ip localhost \
    --robot unitree_g1_with_hands \
    --max_steps ${max_steps} \
    --log_dir ../logs_v6_1_1


