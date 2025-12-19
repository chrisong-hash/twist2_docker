#!/bin/bash

# Teleoperation script with Inspire Hand support
# This script runs the teleoperation system with trigger-based Inspire hand control

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

# Configuration
actual_human_height=1.65
redis_ip="localhost"

# Inspire hand IPs
inspire_left_ip="192.168.123.210"
inspire_right_ip="192.168.123.211"

# Run teleoperation with Inspire hands
python xrobot_teleop_inspire.py \
    --robot unitree_g1 \
    --actual_human_height $actual_human_height \
    --redis_ip $redis_ip \
    --target_fps 100 \
    --measure_fps 1 \
    --use_inspire_hands \
    --inspire_left_ip $inspire_left_ip \
    --inspire_right_ip $inspire_right_ip

echo "Teleoperation with Inspire hands stopped."



