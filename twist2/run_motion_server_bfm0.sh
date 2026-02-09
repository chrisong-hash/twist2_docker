#!/bin/bash
# BFM-Zero compatible motion server
# Publishes BOTH mimic_obs (for TWIST2) AND full pose data (for BFM-Zero backward inference)

script_dir=$(dirname $(realpath $0))
motion_file="${script_dir}/assets/example_motions/0807_yanjie_walk_001.pkl"
redis_ip="localhost"

# Parse arguments
LOOP=""
VIS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --loop)
            LOOP="--loop"
            shift
            ;;
        --vis)
            VIS="--vis"
            shift
            ;;
        --motion_file)
            motion_file="$2"
            shift 2
            ;;
        --redis_ip)
            redis_ip="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

cd deploy_real

echo "========================================"
echo "BFM-Zero Motion Server"
echo "========================================"
echo "Motion file: ${motion_file}"
echo "Redis IP: ${redis_ip}"
echo "Loop: ${LOOP:-disabled}"
echo "Vis: ${VIS:-disabled}"
echo ""
echo "Publishing to Redis:"
echo "  - action_body_unitree_g1_with_hands (mimic_obs format)"
echo "  - bfm_motion_unitree_g1_with_hands (full pose for BFM)"
echo "========================================"

python server_motion_lib_bfm.py \
    --motion_file ${motion_file} \
    --robot unitree_g1_with_hands \
    --redis_ip ${redis_ip} \
    ${LOOP} \
    ${VIS}

