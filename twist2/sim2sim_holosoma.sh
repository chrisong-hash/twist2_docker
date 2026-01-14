#!/bin/bash
# sim2sim with Holosoma natural gait policy
#
# This uses the Holosoma V3 policy for upright, natural walking
# instead of RoboMimic's crouched walking
#
# Usage:
#   bash sim2sim_holosoma.sh
#
# Controls:
#   - Send velocity commands via Redis key "loco_vel_cmd" as [vx, vy, yaw_rate]
#   - Or run hybrid_loco_teleop.py to control via Pico joystick

SCRIPT_DIR=$(dirname $(realpath $0))

# TWIST2 upper body policy
twist2_policy=${SCRIPT_DIR}/assets/ckpts/twist2_1017_20k.onnx

# Holosoma natural gait policy for legs
holosoma_policy=${SCRIPT_DIR}/assets/ckpts/holosoma_natural_gait_v3.onnx

# Check if Holosoma model exists
if [ ! -f "$holosoma_policy" ]; then
    echo "ERROR: Holosoma model not found at $holosoma_policy"
    echo ""
    echo "Please copy the model first:"
    echo "  cp /home/robo/CodeSpace/natural_gait_g1/trained_models/v3_stable/model_0050000.onnx \\"
    echo "     ${holosoma_policy}"
    exit 1
fi

cd deploy_real

echo "=============================================="
echo "  sim2sim with HOLOSOMA Natural Gait"
echo "=============================================="
echo ""
echo "Upper body: TWIST2 (${twist2_policy})"
echo "Legs:       Holosoma V3 (${holosoma_policy})"
echo ""
echo "To send walking commands, run in another terminal:"
echo "  redis-cli SET loco_vel_cmd '[0.3, 0.0, 0.0]'  # Walk forward"
echo "  redis-cli SET loco_vel_cmd '[0.0, 0.0, 0.0]'  # Stop"
echo ""
echo "Or use hybrid_loco_teleop.py with Pico controller"
echo "=============================================="
echo ""

python server_low_level_g1_sim.py \
    --xml ../assets/g1/g1_sim2sim_29dof.xml \
    --policy ${twist2_policy} \
    --device cuda \
    --measure_fps 1 \
    --policy_frequency 50 \
    --limit_fps 1 \
    --hybrid_loco_mode \
    --use_holosoma \
    --holosoma_model ${holosoma_policy}

