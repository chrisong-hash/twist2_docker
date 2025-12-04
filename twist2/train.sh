#!/bin/bash

# Usage: bash train.sh <experiment_id> <device>

# bash train.sh 1103_twist2 cuda:0


cd legged_gym/legged_gym/scripts

robot_name="g1"
exptid=$1
device=$2

# For testing without teacher model, use priv_mimic (PPO)
# For full training with teacher, use stu_future (DAgger)
task_name="${robot_name}_priv_mimic"
proj_name="${robot_name}_priv_mimic"


# Run the training script
python train.py --task "${task_name}" \
                --proj_name "${proj_name}" \
                --exptid "${exptid}" \
                --device "${device}" \
                --teacher_exptid "None" \
                # --resume \
                # --debug \
