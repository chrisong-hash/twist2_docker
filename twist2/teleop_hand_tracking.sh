#!/bin/bash

# Teleoperation with Pico Hand Tracking + Inspire Hands
# Full body teleop with actual finger tracking from Pico VR

# Make sure Redis is running
redis-cli ping > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Starting Redis server..."
    redis-server --daemonize yes
    sleep 1
fi

# Navigate to deployment directory
cd "$(dirname "$0")/deploy_real" || exit 1

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate gmr

echo "=============================================="
echo "🤖 TELEOP WITH HAND TRACKING"
echo "=============================================="
echo ""
echo "Features:"
echo "  🖐️  Pico hand tracking → Inspire hands"
echo "  🦾  Full body tracking → Robot"
echo "  👏  Clap to pause/resume (safety)"
echo ""

# Configuration
actual_human_height=1.65
redis_ip="localhost"

# Inspire hand IPs
inspire_left_ip="192.168.123.210"
inspire_right_ip="192.168.123.211"

# Parse arguments
USE_HANDS=""
SKIP_CALIB=""
for arg in "$@"; do
    case $arg in
        --hands)
            USE_HANDS="--use_inspire_hands"
            echo "✓ Inspire hands ENABLED"
            ;;
        --skip-calibration|-s)
            SKIP_CALIB="--skip_calibration"
            echo "✓ Calibration SKIPPED (using defaults)"
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --hands              Enable Inspire hand control"
            echo "  --skip-calibration   Skip hand calibration"
            echo "  --help               Show this help"
            exit 0
            ;;
    esac
done

if [ -z "$USE_HANDS" ]; then
    echo "ℹ  Running WITHOUT Inspire hands (use --hands to enable)"
fi

echo ""
echo "Make sure:"
echo "  1. XRoboToolkit is running on PC"
echo "  2. Pico is connected to PC's hotspot"
echo "  3. sim2real is running on robot (if controlling robot)"
echo ""
echo "Starting..."
echo ""

# Run teleoperation with hand tracking
python xrobot_teleop_hand_tracking.py \
    --robot unitree_g1 \
    --actual_human_height $actual_human_height \
    --redis_ip $redis_ip \
    --target_fps 100 \
    --measure_fps 1 \
    $USE_HANDS \
    $SKIP_CALIB \
    --inspire_left_ip $inspire_left_ip \
    --inspire_right_ip $inspire_right_ip

echo "Teleoperation stopped."

