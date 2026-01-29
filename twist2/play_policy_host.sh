#!/bin/bash
# play_policy_host.sh - Visualize teacher policy directly on HOST (not Docker)
#
# Usage: ./play_policy_host.sh [checkpoint] [motion_config]
# Example: ./play_policy_host.sh 12500
# Example: ./play_policy_host.sh 12500 single   # Uses same motion as run_motion_server.sh
# Example: ./play_policy_host.sh 12500 example  # Uses example_motions_host.yaml
#
# Requirements:
# - conda environment 'twist2' with Isaac Gym dependencies
# - Isaac Gym in ~/Downloads/isaacgym

set -e

CHECKPOINT=${1:-20000}
MOTION_CONFIG=${2:-single}  # Default to single motion (matches motion server)
PROJ_NAME="h1"
EXPTID="responsive_v6"
TASK="g1_priv_mimic"

# Select motion config file
case "$MOTION_CONFIG" in
    single)
        MOTION_YAML="single_motion_host.yaml"  # Same motion as run_motion_server.sh
        ;;
    example)
        MOTION_YAML="example_motions_host.yaml"
        ;;
    *)
        MOTION_YAML="$MOTION_CONFIG"  # Allow custom yaml filename
        ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TWIST2_DIR="$SCRIPT_DIR"
ISAACGYM_DIR="$HOME/Downloads/isaacgym/python"

echo "============================================"
echo "  TWIST2 Policy Visualizer (Host)"
echo "============================================"
echo "  Task: $TASK"
echo "  Project: $PROJ_NAME"
echo "  Experiment: $EXPTID"
echo "  Checkpoint: $CHECKPOINT"
echo "  Motion Config: $MOTION_YAML"
echo "============================================"

# Check Isaac Gym
if [ ! -d "$ISAACGYM_DIR" ]; then
    echo "Error: Isaac Gym not found at $ISAACGYM_DIR"
    echo "Please install Isaac Gym or modify ISAACGYM_DIR in this script."
    exit 1
fi

# Temporarily swap to host-compatible motion config
CONFIG_FILE="$TWIST2_DIR/legged_gym/legged_gym/envs/g1/g1_mimic_distill_config.py"
BACKUP_FILE="/tmp/g1_mimic_distill_config.py.bak"

echo ""
echo "[1/4] Backing up config and switching to host-compatible motion file..."
cp "$CONFIG_FILE" "$BACKUP_FILE"
# Replace any motion_file path with our selected motion config
sed -i 's|motion_file = f"{LEGGED_GYM_ROOT_DIR}/motion_data_configs/[^"]*"|motion_file = f"{LEGGED_GYM_ROOT_DIR}/motion_data_configs/'"$MOTION_YAML"'"|g' "$CONFIG_FILE"

# Function to restore config on exit
cleanup() {
    echo ""
    echo "[4/4] Restoring original config..."
    cp "$BACKUP_FILE" "$CONFIG_FILE"
    rm -f "$BACKUP_FILE"
}
trap cleanup EXIT

echo "[2/4] Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate twist2

echo "[3/4] Running visualization..."
echo ""

export LD_LIBRARY_PATH="$HOME/anaconda3/envs/twist2/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$ISAACGYM_DIR:$TWIST2_DIR/legged_gym:$TWIST2_DIR/rsl_rl:$TWIST2_DIR/pose:$PYTHONPATH"

cd "$TWIST2_DIR/legged_gym/legged_gym/scripts"

python play.py \
    --task "$TASK" \
    --resume \
    --load_run "$EXPTID" \
    --checkpoint "$CHECKPOINT" \
    --proj_name "$PROJ_NAME" \
    --exptid "$EXPTID"

