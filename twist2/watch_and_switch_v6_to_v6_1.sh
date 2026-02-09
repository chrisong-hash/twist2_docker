#!/bin/bash
#
# Watch for V6 to reach 20k iterations, then switch to V6.1
#
# Usage: bash watch_and_switch_v6_to_v6_1.sh
#
# This script:
#   1. Monitors for model_20000.pt to appear
#   2. Gracefully stops V6 training
#   3. Starts V6.1 training automatically
#

V6_CHECKPOINT="/workspace/twist2/legged_gym/logs/h1/responsive_v6/model_20000.pt"
CHECK_INTERVAL=60  # Check every 60 seconds

echo "=============================================="
echo "  V6 → V6.1 Auto-Switch Monitor"
echo "=============================================="
echo ""
echo "Watching for: $V6_CHECKPOINT"
echo "Check interval: ${CHECK_INTERVAL}s"
echo ""

# Find V6 process
V6_PID=$(pgrep -f "responsive_v6.*train.py" || pgrep -f "g1_priv_mimic_v6.*train.py")
if [ -z "$V6_PID" ]; then
    echo "⚠️  Warning: V6 process not found. Will still watch for checkpoint."
else
    echo "V6 PID: $V6_PID"
fi
echo ""

# Wait for checkpoint
echo "Waiting for V6 to reach 20k iterations..."
while [ ! -f "$V6_CHECKPOINT" ]; do
    # Check current progress
    LATEST=$(ls -t /workspace/twist2/legged_gym/logs/h1/responsive_v6/model_*.pt 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then
        ITER=$(basename "$LATEST" | sed 's/model_\([0-9]*\).pt/\1/')
        echo "$(date '+%H:%M:%S'): Current checkpoint: model_${ITER}.pt (waiting for 20000...)"
    else
        echo "$(date '+%H:%M:%S'): No checkpoint found yet..."
    fi
    sleep $CHECK_INTERVAL
done

echo ""
echo "=============================================="
echo "  🎉 V6 reached 20k iterations!"
echo "=============================================="
echo ""

# Gracefully stop V6
if [ -n "$V6_PID" ] && ps -p $V6_PID > /dev/null 2>&1; then
    echo "Stopping V6 (PID: $V6_PID)..."
    kill -SIGINT $V6_PID
    sleep 5
    
    # Force kill if still running
    if ps -p $V6_PID > /dev/null 2>&1; then
        echo "Force stopping V6..."
        kill -9 $V6_PID
    fi
    echo "V6 stopped."
else
    echo "V6 process not found or already stopped."
fi

echo ""
sleep 5

# Start V6.1
echo "=============================================="
echo "  Starting V6.1 Training..."
echo "=============================================="
echo ""

cd /workspace/twist2/legged_gym/legged_gym/scripts

python train.py \
    --task g1_priv_mimic_v6_1 \
    --proj_name h1 \
    --exptid responsive_v6_1 \
    --device cuda:0 \
    --num_envs 4096 \
    --headless \
    --wandb_project twist2_walking \
    --wandb_entity chrisong-eastworld

echo ""
echo "V6.1 training complete or interrupted."





