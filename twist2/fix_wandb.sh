#!/bin/bash

# TWIST2 - Fix WandB for offline/disabled mode
# Run this script if WandB gives login errors

SCRIPT_DIR=$(dirname $(realpath $0))
cd "$SCRIPT_DIR"

echo "Fixing WandB configuration..."

# Restore train.py from backup if it exists
if [ -f "legged_gym/legged_gym/scripts/train.py.backup" ]; then
    echo "Restoring train.py from backup..."
    cp legged_gym/legged_gym/scripts/train.py.backup legged_gym/legged_gym/scripts/train.py
fi

# Set environment variables for current session
export WANDB_MODE=disabled
export WANDB_DISABLED=true

echo "✅ WandB disabled for this session"
echo ""
echo "To make permanent, add to your shell profile:"
echo "  export WANDB_MODE=disabled"
echo "  export WANDB_DISABLED=true"
echo ""
echo "Or rebuild the Docker container with the updated Dockerfile"

