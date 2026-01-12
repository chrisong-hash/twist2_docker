#!/bin/bash
# TWIST2 Full Robot Control - runs sim2real + video + neck

# Activate conda environment
source ~/miniconda3/bin/activate twist2

SCRIPT_DIR=$(dirname $(realpath $0))
ckpt_path=${SCRIPT_DIR}/assets/ckpts/twist2_1017_20k.onnx

# Network interface that connects to robot internal network
net=enp4s0

# Redis server (PC IP where teleop runs)
REDIS_HOST=${REDIS_HOST:-192.168.50.164}

# Neck calibration - ADJUST THESE FOR YOUR ASSEMBLY
YAW_CENTER=${YAW_CENTER:-2048}
PITCH_CENTER=${PITCH_CENTER:-2048}

# Dynamixel device
DYNAMIXEL_DEV=${DYNAMIXEL_DEV:-/dev/ttyUSB0}

echo "=============================================="
echo "  TWIST2 Full Robot Control"
echo "=============================================="
echo "  Redis:        $REDIS_HOST"
echo "  Yaw center:   $YAW_CENTER"
echo "  Pitch center: $PITCH_CENTER"
echo "  Dynamixel:    $DYNAMIXEL_DEV"
echo "=============================================="

cd ${SCRIPT_DIR}/deploy_real

# Start peripherals (video + neck) in background
echo ""
echo "🚀 Starting peripherals (video + neck)..."
python robot_peripherals.py \
    --redis ${REDIS_HOST} \
    --yaw_center ${YAW_CENTER} \
    --pitch_center ${PITCH_CENTER} \
    --device ${DYNAMIXEL_DEV} &

PERIPHERALS_PID=$!
echo "   Peripherals PID: $PERIPHERALS_PID"

# Give peripherals time to initialize
sleep 2

# Start main robot control
echo ""
echo "🤖 Starting main robot control..."
python server_low_level_g1_real.py \
    --policy ${ckpt_path} \
    --net ${net} \
    --device cuda \
    --use_hand \
    --smooth_body 0.5

# When main control exits, stop peripherals
echo ""
echo "🛑 Stopping peripherals..."
kill $PERIPHERALS_PID 2>/dev/null
wait $PERIPHERALS_PID 2>/dev/null

echo "Done."


