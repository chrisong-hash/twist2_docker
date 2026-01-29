#!/bin/bash
# Motion Server V6.2 - Publishes current + 3 future frames (0.1s, 0.3s, 0.5s ahead)
#
# For use with g1_stu_future_v6_2 student policy
#
# Usage: bash run_motion_server_v6_2.sh [motion_file]

script_dir=$(dirname $(realpath $0))
motion_file="${1:-${script_dir}/assets/example_motions/0807_yanjie_walk_001.pkl}"

cd deploy_real

redis_ip="localhost"

echo "=============================================="
echo "  Motion Server V6.2 (0.5s Future Sight)"
echo "=============================================="
echo "  Motion file: ${motion_file}"
echo "  Future frames: [5, 15, 25] steps"
echo "  Future times: [0.1s, 0.3s, 0.5s]"
echo "  Redis keys:"
echo "    - action_body_{robot}: Current frame (35 dims)"
echo "    - action_mimic_future_{robot}: Future frames (105 dims)"
echo "=============================================="

python server_motion_lib_v6_2.py \
    --motion_file ${motion_file} \
    --robot unitree_g1_with_hands \
    --vis \
    --redis_ip ${redis_ip}

