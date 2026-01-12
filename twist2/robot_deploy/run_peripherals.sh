#!/bin/bash
# TWIST2 Robot Peripherals - Video streaming + Neck control
# Run this on the robot

# Check serial port access
if [ -e "/dev/ttyUSB0" ] && [ ! -w "/dev/ttyUSB0" ]; then
    echo "⚠️  No write access to /dev/ttyUSB0"
    echo "   Fix with: sudo chmod 666 /dev/ttyUSB0"
    echo "   Or add user to dialout: sudo usermod -a -G dialout \$USER"
    echo ""
    echo "   Fixing temporarily..."
    sudo chmod 666 /dev/ttyUSB0
fi

SCRIPT_DIR=$(dirname $(realpath $0 2>/dev/null) 2>/dev/null || echo ".")

# Redis server (PC IP where teleop runs)
REDIS_HOST=${REDIS_HOST:-192.168.50.164}

# Neck calibration - calibrated values
YAW_CENTER=${YAW_CENTER:-1338}
PITCH_CENTER=${PITCH_CENTER:-695}

# Dynamixel device
DYNAMIXEL_DEV=${DYNAMIXEL_DEV:-/dev/ttyUSB0}

echo "=============================================="
echo "  TWIST2 Robot Peripherals"
echo "=============================================="
echo "  Redis:        $REDIS_HOST"
echo "  Yaw center:   $YAW_CENTER"
echo "  Pitch center: $PITCH_CENTER"
echo "  Dynamixel:    $DYNAMIXEL_DEV"
echo "=============================================="
echo ""
echo "Usage:"
echo "  ./run_peripherals.sh"
echo "  REDIS_HOST=192.168.1.100 ./run_peripherals.sh"
echo "  YAW_CENTER=2100 PITCH_CENTER=2000 ./run_peripherals.sh"
echo ""

# Run from wherever the script is located
cd "$(dirname "$0")" 2>/dev/null || cd ~

python3 robot_peripherals.py \
    --redis ${REDIS_HOST} \
    --yaw_center ${YAW_CENTER} \
    --pitch_center ${PITCH_CENTER} \
    --device ${DYNAMIXEL_DEV}

