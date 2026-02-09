#!/bin/bash
# sim2real_v6.sh - Deploy V6 student policy to real robot

source ~/miniconda3/bin/activate twist2

SCRIPT_DIR=$(dirname $(realpath $0))

# V6 student policy (trained for responsive teleop)
ckpt_path=${SCRIPT_DIR}/legged_gym/logs/h1/student_from_v6/model_17500.onnx

# Verify checkpoint exists
if [ ! -f "$ckpt_path" ]; then
    echo "ERROR: V6 student checkpoint not found at: $ckpt_path"
    echo "Please ensure student distillation training is complete."
    exit 1
fi

# change the network interface name to your own that connects to the robot
# net=enp0s31f6
net=enp4s0

# Clear stale Redis keys from previous sessions to prevent spasm on startup
echo "Clearing stale Redis keys..."
redis-cli DEL action_body_unitree_g1_with_hands action_hand_left_unitree_g1_with_hands action_hand_right_unitree_g1_with_hands action_neck_unitree_g1_with_hands teleop_state_info loco_vel_cmd t_action sim2real_ready > /dev/null 2>&1
echo "Redis keys cleared."

echo "=========================================="
echo "  TWIST2 V6 Sim2Real Deployment"
echo "=========================================="
echo "  Policy: V6 Student (Responsive Teleop)"
echo "  Checkpoint: $ckpt_path"
echo "  Network: $net"
echo "=========================================="

cd deploy_real

python server_low_level_g1_real.py \
    --policy ${ckpt_path} \
    --net ${net} \
    --device cuda \
    --use_hand \
    --smooth_body 0.5
    # --record_proprio \





