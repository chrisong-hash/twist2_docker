#!/bin/bash
# Sim2Sim V6.1.1 - Pure L2 Loss (same obs structure as V6.2)
#
# V6.1.1 uses the same observation structure as V6.2 (1502 dims with future obs)
# but is trained with pure L2 loss instead of KL divergence.
#
# Requires:
#   1. Motion Server (run_motion_server_raw.sh) publishing actual future observations
#   2. V6.1.1 student ONNX policy
#
# Usage: bash sim2sim_v6_1_1.sh [policy_path]

SCRIPT_DIR=$(dirname $(realpath $0))

# Default to V6.1.1 student policy (in workspace for Docker access)
# Note: Using original name because ONNX references model_17500.onnx.data internally
ckpt_path=${1:-${SCRIPT_DIR}/model_17500.onnx}

echo "=============================================="
echo "  Sim2Sim V6.1.1 (Pure L2 Loss)"
echo "=============================================="
echo "  Policy: ${ckpt_path}"
echo "  Expected obs size: 1502 dims"
echo "    - Current: 127 dims (35 mimic + 92 proprio)"
echo "    - History: 1270 dims (10 × 127)"
echo "    - Future: 105 dims (3 × 35)"
echo ""
echo "  Make sure Raw Motion Server is running!"
echo "  (bash run_motion_server_raw.sh)"
echo "=============================================="

cd deploy_real

python server_low_level_g1_sim_v6_2.py \
    --xml ../assets/g1/g1_sim2sim_29dof.xml \
    --policy ${ckpt_path} \
    --device cuda \
    --redis_ip localhost \
    --robot unitree_g1_with_hands

