#!/bin/bash
# Sim2Sim V6.7 - HYBRID: Default + V6.3
#
# Switches between two policies based on commanded motion:
# - Standing (low velocity): Uses DEFAULT policy (stable)
# - Moving (high velocity): Uses V6.3 policy (better tracking)
#
# Usage: bash sim2sim_v6_7.sh

SCRIPT_DIR=$(dirname $(realpath $0))

# Default policy (V6 student) - good at standing still
default_policy=${SCRIPT_DIR}/assets/ckpts/twist2_v6_student.onnx

# Tracking policy (V6.3) - good at motion tracking
tracking_policy=${SCRIPT_DIR}/legged_gym/logs/g1_priv_mimic/v6_3_fixed/model_27500.onnx

# Check policies exist
if [ ! -f "$default_policy" ]; then
    echo "ERROR: Default policy not found: $default_policy"
    exit 1
fi

if [ ! -f "$tracking_policy" ]; then
    echo "ERROR: Tracking policy not found: $tracking_policy"
    echo "Make sure V6.3 model exists at: $tracking_policy"
    exit 1
fi

echo "=============================================="
echo "  Sim2Sim V6.7 - HYBRID MODE"
echo "=============================================="
echo ""
echo "  POLICIES:"
echo "    Standing: $default_policy"
echo "    Tracking: $tracking_policy"
echo ""
echo "  BEHAVIOR:"
echo "    - Low velocity commands → DEFAULT (stable standing)"
echo "    - High velocity commands → V6.3 (better tracking)"
echo ""
echo "  Velocity threshold: 0.05"
echo "=============================================="

cd deploy_real

python server_low_level_g1_sim_v6_7.py \
    --xml ../assets/g1/g1_sim2sim_29dof.xml \
    --default_policy ${default_policy} \
    --tracking_policy ${tracking_policy} \
    --device cuda \
    --redis_ip localhost \
    --robot unitree_g1_with_hands


