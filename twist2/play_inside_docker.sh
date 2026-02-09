#!/bin/bash
# play_inside_docker.sh - Visualize teacher policy (run from INSIDE Docker)
#
# Usage: ./play_inside_docker.sh [checkpoint]
# Example: ./play_inside_docker.sh 12500

CHECKPOINT=${1:-12500}
PROJ_NAME="h1"
EXPTID="responsive_v6"
TASK="g1_priv_mimic"

echo "============================================"
echo "  TWIST2 Policy Visualizer (Inside Docker)"
echo "============================================"
echo "  Task: $TASK"
echo "  Project: $PROJ_NAME"
echo "  Experiment: $EXPTID"
echo "  Checkpoint: $CHECKPOINT"
echo "============================================"
echo ""

cd /workspace/twist2/legged_gym/legged_gym/scripts

python play.py \
    --task "$TASK" \
    --resume \
    --load_run "$EXPTID" \
    --checkpoint "$CHECKPOINT" \
    --proj_name "$PROJ_NAME" \
    --exptid "$EXPTID"





