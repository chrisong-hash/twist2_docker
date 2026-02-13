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

# Config file location
CONFIG_FILE="${SCRIPT_DIR}/neck_config.json"

# Default values
DEFAULT_YAW_CENTER=2048
DEFAULT_PITCH_CENTER=2048
DEFAULT_REDIS_HOST="192.168.50.164"
DEFAULT_DYNAMIXEL_DEV="/dev/ttyUSB0"

# Read from config file if it exists
if [ -f "$CONFIG_FILE" ]; then
    echo "[Config] Reading from $CONFIG_FILE"
    # Use python to parse JSON (more reliable than jq which may not be installed)
    CONFIG_YAW=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE')).get('yaw_center', $DEFAULT_YAW_CENTER))" 2>/dev/null)
    CONFIG_PITCH=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE')).get('pitch_center', $DEFAULT_PITCH_CENTER))" 2>/dev/null)
    CONFIG_REDIS=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE')).get('redis_host', '$DEFAULT_REDIS_HOST'))" 2>/dev/null)
    CONFIG_DEV=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE')).get('dynamixel_dev', '$DEFAULT_DYNAMIXEL_DEV'))" 2>/dev/null)
else
    echo "[Config] No config file found at $CONFIG_FILE"
    echo "[Config] Using defaults (create config with: ./run_peripherals.sh --save-config)"
    CONFIG_YAW=$DEFAULT_YAW_CENTER
    CONFIG_PITCH=$DEFAULT_PITCH_CENTER
    CONFIG_REDIS=$DEFAULT_REDIS_HOST
    CONFIG_DEV=$DEFAULT_DYNAMIXEL_DEV
fi

# Environment variables override config file
REDIS_HOST=${REDIS_HOST:-$CONFIG_REDIS}
YAW_CENTER=${YAW_CENTER:-$CONFIG_YAW}
PITCH_CENTER=${PITCH_CENTER:-$CONFIG_PITCH}
DYNAMIXEL_DEV=${DYNAMIXEL_DEV:-$CONFIG_DEV}

# Handle --save-config option
if [ "$1" = "--save-config" ]; then
    echo "[Config] Saving configuration to $CONFIG_FILE"
    cat > "$CONFIG_FILE" << EOF
{
    "yaw_center": ${YAW_CENTER},
    "pitch_center": ${PITCH_CENTER},
    "redis_host": "${REDIS_HOST}",
    "dynamixel_dev": "${DYNAMIXEL_DEV}"
}
EOF
    echo "[Config] Saved:"
    cat "$CONFIG_FILE"
    echo ""
    echo "Config file created. Run without --save-config to start peripherals."
    exit 0
fi

echo "=============================================="
echo "  TWIST2 Robot Peripherals"
echo "=============================================="
echo "  Config:       $CONFIG_FILE"
echo "  Redis:        $REDIS_HOST"
echo "  Yaw center:   $YAW_CENTER"
echo "  Pitch center: $PITCH_CENTER"
echo "  Dynamixel:    $DYNAMIXEL_DEV"
echo "=============================================="
echo ""
echo "Usage:"
echo "  ./run_peripherals.sh                     # Run with config/defaults"
echo "  ./run_peripherals.sh --save-config       # Save current settings to config"
echo ""
echo "  # Override and save new calibration:"
echo "  YAW_CENTER=3064 PITCH_CENTER=3700 ./run_peripherals.sh --save-config"
echo ""

# Run from wherever the script is located
cd "$(dirname "$0")" 2>/dev/null || cd ~

python3 robot_peripherals.py \
    --redis ${REDIS_HOST} \
    --yaw_center ${YAW_CENTER} \
    --pitch_center ${PITCH_CENTER} \
    --device ${DYNAMIXEL_DEV}


    cat > "$CONFIG_FILE" << EOF
{
    "yaw_center": ${YAW_CENTER},
    "pitch_center": ${PITCH_CENTER},
    "redis_host": "${REDIS_HOST}",
    "dynamixel_dev": "${DYNAMIXEL_DEV}"
}
EOF
    echo "[Config] Saved:"
    cat "$CONFIG_FILE"
    echo ""
    echo "Config file created. Run without --save-config to start peripherals."
    exit 0
fi

echo "=============================================="
echo "  TWIST2 Robot Peripherals"
echo "=============================================="
echo "  Config:       $CONFIG_FILE"
echo "  Redis:        $REDIS_HOST"
echo "  Yaw center:   $YAW_CENTER"
echo "  Pitch center: $PITCH_CENTER"
echo "  Dynamixel:    $DYNAMIXEL_DEV"
echo "=============================================="
echo ""
echo "Usage:"
echo "  ./run_peripherals.sh                     # Run with config/defaults"
echo "  ./run_peripherals.sh --save-config       # Save current settings to config"
echo ""
echo "  # Override and save new calibration:"
echo "  YAW_CENTER=3064 PITCH_CENTER=3700 ./run_peripherals.sh --save-config"
echo ""

# Run from wherever the script is located
cd "$(dirname "$0")" 2>/dev/null || cd ~

python3 robot_peripherals.py \
    --redis ${REDIS_HOST} \
    --yaw_center ${YAW_CENTER} \
    --pitch_center ${PITCH_CENTER} \
    --device ${DYNAMIXEL_DEV}
