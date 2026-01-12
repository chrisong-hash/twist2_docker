#!/bin/bash
# Deploy robot peripherals scripts to Unitree robot
# Usage: ./deploy_to_robot.sh [robot_ip]

ROBOT_USER="unitree"
ROBOT_IP="${1:-192.168.123.164}"
ROBOT_DEST="/home/unitree"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================"
echo "  Deploying to ${ROBOT_USER}@${ROBOT_IP}"
echo "========================================"

# Files to deploy
FILES=(
    "robot_peripherals.py"
    "read_neck_position.py"
    "run_peripherals.sh"
    "reset_zed_usb.sh"
)

# Copy files
for file in "${FILES[@]}"; do
    if [ -f "${SCRIPT_DIR}/${file}" ]; then
        echo "Copying ${file}..."
        scp "${SCRIPT_DIR}/${file}" "${ROBOT_USER}@${ROBOT_IP}:${ROBOT_DEST}/"
    else
        echo "WARNING: ${file} not found!"
    fi
done

echo ""
echo "========================================"
echo "  Deployment complete!"
echo "========================================"
echo ""
echo "On the robot, run:"
echo "  cd ~ && ./run_peripherals.sh"
echo ""
echo "Or manually:"
echo "  python3 robot_peripherals.py --redis <PC_IP>"

