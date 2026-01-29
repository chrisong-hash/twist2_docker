#!/bin/bash
# Record motion from PICO/XRobo teleop to pkl files for training lower level policy
#
# Usage: bash record_motion.sh [prefix]
# Example: bash record_motion.sh sk_walk
#          bash record_motion.sh sk_jump

cd deploy_real

# Default prefix is c_walk, can be overridden by first argument
PREFIX=${1:-c_walk}

# Human height (adjust based on the operator)
HUMAN_HEIGHT=1.8

python record_motion_pkl.py \
    --robot unitree_g1 \
    --output_dir /workspace/data/virtuals_dataset/c_walk \
    --prefix $PREFIX \
    --actual_human_height $HUMAN_HEIGHT \
    --target_fps 30
