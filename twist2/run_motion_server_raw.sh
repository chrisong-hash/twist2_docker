#!/bin/bash
# Raw Motion Server - Publishes to motion_raw_* for buffered pipeline
#
# Use this WITH the buffered server for V6.2:
#   Terminal 1: bash run_motion_server_raw.sh      # Publishes to motion_raw_*
#   Terminal 2: bash run_motion_server_v6_2_buffered.sh  # Reads motion_raw_*, publishes action_body_*
#   Terminal 3: bash sim2sim_v6_2.sh               # Reads action_body_*
#
# This creates the time-delay buffering pipeline for V6.2 "future sight"

SCRIPT_DIR=$(dirname $(realpath $0))

echo "=============================================="
echo "  Raw Motion Server (for V6.2 buffered pipeline)"
echo "=============================================="
echo "  Publishes to: motion_raw_* (35 dims)"
echo ""
echo "  Run buffered server in another terminal:"
echo "    bash run_motion_server_v6_2_buffered.sh"
echo "=============================================="

cd ${SCRIPT_DIR}/deploy_real

# Use direct pkl file like original motion server
motion_file="${SCRIPT_DIR}/assets/example_motions/0807_yanjie_walk_003.pkl"

python server_motion_lib_raw.py \
    --robot unitree_g1_with_hands \
    --vis \
    --motion_file ${motion_file}

