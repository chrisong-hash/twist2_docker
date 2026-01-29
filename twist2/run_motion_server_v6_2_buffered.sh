#!/bin/bash
# V6.2 Buffered Motion Server - Provides "future" through time-delay
#
# PIPELINE (3 terminals):
#   1. bash run_motion_server_raw.sh           # Publishes motion_raw_*
#   2. bash run_motion_server_v6_2_buffered.sh # Reads motion_raw_*, publishes action_body_*
#   3. bash sim2sim_v6_2.sh                    # Reads action_body_*, runs robot
#
# Usage: bash run_motion_server_v6_2_buffered.sh [redis_ip]

SCRIPT_DIR=$(dirname $(realpath $0))
REDIS_IP=${1:-localhost}

echo "=============================================="
echo "  V6.2 Buffered Motion Server"
echo "=============================================="
echo "  Buffer: 25 frames (0.5s at 50Hz)"
echo "  Latency: 0.5s response delay"
echo "  Future frames: +0.1s, +0.3s, +0.5s"
echo ""
echo "  Reads from: motion_raw_* (from raw motion server)"
echo "  Writes to:  action_body_* (delayed 0.5s)"
echo "              action_mimic_future_* (105 dims)"
echo "=============================================="

cd ${SCRIPT_DIR}/deploy_real

python server_motion_lib_v6_2_buffered.py \
    --redis_ip ${REDIS_IP} \
    --robot unitree_g1_with_hands

