#!/bin/bash
# Looping Motion Server
# Continuously streams motion data to Redis for BFM-Zero testing
# Uses internal loop (no Python restart, single MuJoCo viewer)

script_dir=$(dirname $(realpath $0))
motion_file="${script_dir}/assets/example_motions/0807_yanjie_walk_001.pkl"

# Change to deploy_real directory
cd deploy_real

# Redis server
redis_ip="localhost"

echo "=============================================="
echo " Motion Server Loop (Internal Loop Mode)"
echo " Motion: ${motion_file}"
echo " Redis: ${redis_ip}"
echo " Press Ctrl+C to stop"
echo "=============================================="

# Run the motion server with --loop flag
# This loops internally without restarting Python
python server_motion_lib.py \
    --motion_file ${motion_file} \
    --robot unitree_g1_with_hands \
    --redis_ip ${redis_ip} \
    --vis \
    --loop

echo "[Motion Server] Stopped."
