#!/bin/bash
# BFM-Zero Teleop Server
# Reads from PICO VR and publishes full pose to Redis for BFM-Zero

cd "$(dirname "$0")/deploy_real"

echo "========================================"
echo "  BFM-Zero Teleop Server"
echo "========================================"
echo ""
echo "This script:"
echo "  1. Connects to PICO VR via XRobotStreamer"
echo "  2. Uses GMR to retarget to full robot pose"
echo "  3. Publishes to Redis:"
echo "     - bfm_motion_unitree_g1_with_hands (full pose)"
echo "     - controller_unitree_g1_with_hands (PICO buttons)"
echo ""
echo "Usage:"
echo "  bash teleop_bfm.sh [--no-vis] [--redis_ip IP]"
echo ""
echo "  --no-vis    Disable MuJoCo visualization (enabled by default)"
echo "  --redis_ip  Redis server IP (default: localhost)"
echo ""
echo "========================================"
echo ""

# Activate gmr conda environment
echo "Activating gmr conda environment..."
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate gmr 2>/dev/null || echo "Note: gmr env not found, using current env"
echo "Python: $(which python)"
echo ""

# Parse args
VIS="--vis"  # Default: visualization enabled
REDIS_IP="localhost"

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-vis)
            VIS=""
            echo "MuJoCo visualization: DISABLED"
            shift
            ;;
        --redis_ip)
            REDIS_IP="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

echo "Redis IP: ${REDIS_IP}"
echo ""

python teleop_bfm.py --robot unitree_g1_with_hands --redis_ip ${REDIS_IP} ${VIS}
