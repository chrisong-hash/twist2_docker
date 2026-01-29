SCRIPT_DIR=$(dirname $(realpath $0))
# ckpt_path=${SCRIPT_DIR}/assets/ckpts/twist2_1017_20k.onnx
ckpt_path=${SCRIPT_DIR}/assets/ckpts/twist2_v6_student.onnx   # Original V6 student
# ckpt_path=/home/robo/CodeSpace/twist2_docker/twist2/legged_gym/logs/h1/student_v6_2/model_6000.onnx
# ckpt_path=${SCRIPT_DIR}/legged_gym/logs/h1/student_from_v6/model_17500.onnx

cd deploy_real

python server_low_level_g1_sim.py \
    --xml ../assets/g1/g1_sim2sim_29dof.xml \
    --policy ${ckpt_path} \
    --device cuda \
    --measure_fps 1 \
    --policy_frequency 100 \
    --limit_fps 1 \
    # --record_proprio \
