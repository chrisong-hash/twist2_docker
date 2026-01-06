#!/bin/bash
# ==============================================
# Local Neck Controller for TWIST2
# ==============================================
# Run this instead of docker_neck.sh when testing
# the neck on your desk (not connected to G1)
#
# Usage:
#   bash run_neck_local.sh
#
# Works with:
#   - teleop.sh
#   - hybrid_teleop.sh
#   - gui.sh (modify to call this instead of SSH)
# ==============================================

echo ""
echo "============================================================"
echo "  TWIST2 Local Neck Controller"
echo "============================================================"
echo ""
echo "This controls the neck motors directly via USB"
echo "Make sure:"
echo "  1. Neck is connected via USB (/dev/ttyUSB0)"
echo "  2. Motor IDs set: Yaw=0, Pitch=1"
echo "  3. Baudrate: 57600 (default)"
echo ""
echo "Press Ctrl+C to stop"
echo "============================================================"
echo ""

# Check for USB device
if [ ! -e /dev/ttyUSB0 ]; then
    echo "WARNING: /dev/ttyUSB0 not found!"
    echo "Available USB devices:"
    ls /dev/ttyUSB* 2>/dev/null || echo "  None found"
    echo ""
    read -p "Enter port to use (or press Enter to exit): " PORT
    if [ -z "$PORT" ]; then
        exit 1
    fi
else
    PORT="/dev/ttyUSB0"
fi

# Grant permission
sudo chmod 777 $PORT 2>/dev/null

# Make sure Redis is running
redis-cli ping > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Starting Redis server..."
    redis-server --daemonize yes
    sleep 1
fi

# Install dependencies if needed
pip show dynamixel-sdk > /dev/null 2>&1 || pip install dynamixel-sdk
pip show redis > /dev/null 2>&1 || pip install redis

# Run the local neck controller
cd "$(dirname "$0")/deploy_real"
python neck_controller_local.py --port $PORT

