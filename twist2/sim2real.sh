#!/bin/bash

source ~/miniconda3/bin/activate twist2

SCRIPT_DIR=$(dirname $(realpath $0))
ckpt_path=${SCRIPT_DIR}/assets/ckpts/twist2_1017_20k.onnx

# change the network interface name to your own that connects to the robot
# net=enp0s31f6
net=enp4s0

# Clear stale Redis keys from previous sessions to prevent spasm on startup
echo "Clearing stale Redis keys..."
redis-cli DEL action_body_unitree_g1_with_hands action_hand_left_unitree_g1_with_hands action_hand_right_unitree_g1_with_hands action_neck_unitree_g1_with_hands teleop_state_info loco_vel_cmd t_action sim2real_ready > /dev/null 2>&1
echo "Redis keys cleared."

cd deploy_real

python server_low_level_g1_real.py \
    --policy ${ckpt_path} \
    --net ${net} \
    --device cuda \
    --use_hand \
    --smooth_body 0.5
    # --record_proprio \
