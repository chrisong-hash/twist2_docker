#!/bin/bash
# Sim2Sim V6.2 - Uses real future observations (0.1s, 0.3s, 0.5s ahead)
#
# Requires:
#   1. Motion Server V6.2 running (bash run_motion_server_v6_2.sh)
#   2. V6.2 student ONNX policy trained with g1_stu_future_v6_2
#
# Usage: bash sim2sim_v6_2.sh [policy_path]

SCRIPT_DIR=$(dirname $(realpath $0))

# Default to V6.2 student policy (update this path after training)
ckpt_path=${1:-${SCRIPT_DIR}/legged_gym/logs/h1/student_v6_2/model_latest.onnx}

echo "=============================================="
echo "  Sim2Sim V6.2 (Real Future Observations)"
echo "=============================================="
echo "  Policy: ${ckpt_path}"
echo "  Expected obs size: 2137 dims"
echo "    - Current: 127 dims (35 mimic + 92 proprio)"
echo "    - History: 1905 dims (15 × 127)"
echo "    - Future: 105 dims (3 × 35)"
echo ""
echo "  Make sure Motion Server V6.2 is running!"
echo "  (bash run_motion_server_v6_2.sh)"
echo "=============================================="

cd deploy_real

python server_low_level_g1_sim_v6_2.py \
    --xml ../assets/g1/g1_sim2sim_29dof.xml \
    --policy ${ckpt_path} \
    --device cuda \
    --redis_ip localhost \
    --robot unitree_g1_with_hands

