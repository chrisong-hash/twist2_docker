#!/bin/bash
# play_policy.sh - Visualize teacher policy inside Docker with proper X11 forwarding
#
# Usage (from HOST): ./play_policy.sh [checkpoint]
# Example: ./play_policy.sh 12500
#
# This script handles X11 forwarding and runs play.py inside Docker

set -e

CHECKPOINT=${1:-12500}
PROJ_NAME="h1"
EXPTID="responsive_v6"
TASK="g1_priv_mimic"

echo "============================================"
echo "  TWIST2 Policy Visualizer (Docker)"
echo "============================================"
echo "  Task: $TASK"
echo "  Project: $PROJ_NAME"
echo "  Experiment: $EXPTID"
echo "  Checkpoint: $CHECKPOINT"
echo "============================================"

# Step 1: Allow X11 access from Docker (run on host)
echo ""
echo "[1/3] Allowing X11 access from Docker..."
xhost +local:docker 2>/dev/null || xhost +local:root 2>/dev/null || echo "Warning: xhost command failed (may already be set)"

# Step 2: Check if Docker container is running
echo "[2/3] Checking Docker container..."
if ! docker ps | grep -q twist2; then
    echo "  Starting Docker container..."
    docker-compose up -d
    sleep 5
fi

# Step 3: Run play.py inside Docker
echo "[3/3] Running visualization..."
echo ""

docker exec -it -e DISPLAY=$DISPLAY twist2 bash -c "
    source /opt/miniconda3/etc/profile.d/conda.sh
    conda activate twist2
    cd /workspace/twist2/legged_gym/legged_gym/scripts
    
    # Set Mesa to software rendering as fallback if needed
    # export LIBGL_ALWAYS_SOFTWARE=1
    
    python play.py \
        --task $TASK \
        --resume \
        --load_run $EXPTID \
        --checkpoint $CHECKPOINT \
        --proj_name $PROJ_NAME \
        --exptid $EXPTID
"

# Revoke X11 access when done (optional, for security)
# xhost -local:docker


