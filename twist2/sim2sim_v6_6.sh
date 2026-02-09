#!/bin/bash
# Sim2Sim V6.6 - SIMPLIFIED: Freeze when no motion data
#
# Uses the default V6 student policy (same as original sim2sim.sh)
# FREEZES physics when no motion server running - robot stays perfectly still
# When motion server is running - uses original behavior exactly
#
# Usage: bash sim2sim_v6_6.sh [policy_path]

SCRIPT_DIR=$(dirname $(realpath $0))

# Default to V6 student policy (same as original sim2sim.sh)
ckpt_path=${1:-${SCRIPT_DIR}/assets/ckpts/twist2_v6_student.onnx}

echo "=============================================="
echo "  Sim2Sim V6.6 - Freeze Mode"
echo "=============================================="
echo "  Policy: ${ckpt_path}"
echo ""
echo "  BEHAVIOR:"
echo "    - NO motion server: Robot FREEZES (stands perfectly still)"
echo "    - WITH motion server: Robot tracks motion like original"
echo ""
echo "  This gives the 'standing still' behavior you observed"
echo "  with the original when motion server stops."
echo "=============================================="

cd deploy_real

python server_low_level_g1_sim_v6_6.py \
    --xml ../assets/g1/g1_sim2sim_29dof.xml \
    --policy ${ckpt_path} \
    --device cuda \
    --redis_ip localhost \
    --robot unitree_g1_with_hands
