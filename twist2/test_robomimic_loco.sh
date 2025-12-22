#!/bin/bash
# Test pure RoboMimic LocoMode deployment
# 
# This runs the RoboMimic deployment directly - no TWIST2 integration
# Uses robot's physical Xbox controller for input
#
# Startup sequence:
#   1. Suspend robot with harness, hold L2+R2
#   2. Run this script
#   3. Press START on controller → position control mode
#   4. Hold R1+A → LocoMode (walking)
#   5. Use joystick to walk
#   6. Press SELECT to exit

echo ""
echo "============================================================"
echo "  PURE ROBOMIMIC LOCOMODE TEST"
echo "============================================================"
echo ""
echo "This tests RoboMimic walking without TWIST2"
echo "Uses robot's physical Xbox controller for input"
echo ""
echo "Prerequisites:"
echo "  1. Robot suspended with harness"
echo "  2. Hold L2+R2 on controller"
echo ""
echo "Controls (robot's physical controller):"
echo "  START     : Enter position control mode"
echo "  R1 + A    : Enter LocoMode (walking)"  
echo "  Joystick  : Walk (ly=forward, lx=strafe, rx=rotate)"
echo "  SELECT    : Exit program"
echo "  F1        : Emergency damping mode"
echo "============================================================"
echo ""

# Check if RoboMimic_Deploy is mounted
if [ ! -d "/workspace/RoboMimic_Deploy" ]; then
    echo "ERROR: RoboMimic_Deploy not mounted!"
    echo "Make sure docker-compose.yml has the mount and restart container"
    exit 1
fi

cd /workspace/RoboMimic_Deploy

# Activate gmr environment (has torch)
eval "$(conda shell.bash hook)"
conda activate gmr

# Install missing dependencies
echo "Installing dependencies..."
pip install onnx onnxruntime cyclonedds -q 2>/dev/null

# Check if unitree_sdk2_python is installed
python -c "import unitree_sdk2py" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing unitree_sdk2_python..."
    cd unitree_sdk2_python
    # Use regular install (not -e) for mounted volumes
    pip install . -q 2>/dev/null || true
    cd ..
fi

# Add RoboMimic paths to PYTHONPATH as fallback
export PYTHONPATH="/workspace/RoboMimic_Deploy:/workspace/RoboMimic_Deploy/unitree_sdk2_python:$PYTHONPATH"

echo "Starting RoboMimic deployment..."
echo ""

python deploy_real/deploy_real.py

