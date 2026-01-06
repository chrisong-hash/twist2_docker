#!/bin/bash
# ==============================================
# Headless Neck Teleop
# ==============================================
# Tracks your head from PICO VR and sends to Redis
# No MuJoCo/display needed!
#
# Usage:
#   bash neck_teleop_headless.sh
# ==============================================

echo ""
echo "============================================================"
echo "  Headless Neck Teleop"
echo "============================================================"
echo ""
echo "Prerequisites:"
echo "  1. XRobotToolkit PC Service running on your computer"
echo "  2. PICO VR connected and XRobot app streaming"
echo ""
echo "This script reads head pose from VR and sends neck"
echo "commands to Redis - no display needed!"
echo "============================================================"
echo ""

# Activate GMR environment
eval "$(conda shell.bash hook)"
conda activate gmr

cd "$(dirname "$0")/deploy_real"

# Run headless neck teleop (--invert_pitch for correct direction)
python neck_teleop_headless.py --invert_pitch "$@"

