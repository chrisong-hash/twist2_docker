#!/bin/bash
# TWIST2 Full Sim2Real - Hybrid locomotion + Peripherals (video + neck)
# 
# This script:
#   1. SSHes to robot to start peripherals (video streaming + neck control)
#   2. Runs hybrid locomotion mode with Inspire hands
#   3. Cleans up on exit
#
# Use with: hybrid_teleop.sh (or teleop_inspire.sh for non-hybrid)

# ============== Configuration ==============
ROBOT_USER="unitree"
ROBOT_IP="192.168.123.164"
PC_IP="192.168.123.222"  # PC IP as seen from robot (for Redis)

# Network interface connecting to robot
NET_INTERFACE="enp4s0"

# SSH options
SCRIPT_DIR=$(dirname $(realpath $0))
if [ -f "${SCRIPT_DIR}/robot_deploy/id_robot" ]; then
    SSH_KEY="${SCRIPT_DIR}/robot_deploy/id_robot"
elif [ -f ~/.ssh/id_robot ]; then
    SSH_KEY=~/.ssh/id_robot
else
    echo "[ERROR] SSH key not found! Copy id_robot to twist2/robot_deploy/ or ~/.ssh/"
    exit 1
fi
SSH_OPTS="-i ${SSH_KEY} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=5"

# ============================================

echo "=============================================="
echo "  TWIST2 Full Sim2Real"
echo "  (Hybrid Locomotion + Video + Neck)"
echo "=============================================="
echo "  Robot:     ${ROBOT_USER}@${ROBOT_IP}"
echo "  PC Redis:  ${PC_IP}"
echo "  Network:   ${NET_INTERFACE}"
echo "=============================================="
echo ""
echo "Controls:"
echo "  Right A        : preview → teleop → pause → teleop..."
echo "  Left X         : Toggle teleop_full ↔ teleop_loco"
echo "  Right A+B      : EMERGENCY SHUTDOWN"
echo "  Joysticks      : Walk/rotate (in teleop_loco mode)"
echo "  Triggers/Grips : Inspire hand control"
echo "=============================================="
echo ""

# Activate conda environment
if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source /opt/conda/etc/profile.d/conda.sh
    conda activate twist2 2>/dev/null || true
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source $HOME/miniconda3/etc/profile.d/conda.sh
    conda activate twist2
fi

CKPT_PATH=${SCRIPT_DIR}/assets/ckpts/twist2_1017_20k.onnx

# Function to start peripherals on robot
start_peripherals() {
    echo "[1/3] Starting peripherals on robot..."
    
    # Create logs directory on robot if it doesn't exist
    echo "    Creating logs directory..."
    ssh -n ${SSH_OPTS} ${ROBOT_USER}@${ROBOT_IP} 'mkdir -p ~/logs' || { echo "[✗] Cannot connect to robot"; exit 1; }
    
    # Kill any existing peripheral process
    echo "    Killing existing processes..."
    ssh -n ${SSH_OPTS} ${ROBOT_USER}@${ROBOT_IP} 'pkill -f robot_peripherals.py 2>/dev/null; exit 0'
    sleep 1
    
    # Start peripherals with logging
    echo "    Launching peripheral script (video + neck)..."
    LOG_FILE="peripheral_\$(date +%Y%m%d_%H%M%S).log"
    ssh -f ${SSH_OPTS} ${ROBOT_USER}@${ROBOT_IP} "cd ~ && python3 robot_peripherals.py --redis ${PC_IP} > ~/logs/${LOG_FILE} 2>&1"
    
    sleep 3
    
    # Verify it started
    echo "    Verifying..."
    if ssh -n ${SSH_OPTS} ${ROBOT_USER}@${ROBOT_IP} 'pgrep -f robot_peripherals.py' > /dev/null 2>&1; then
        echo "[✓] Peripherals started (video + neck active)"
    else
        echo "[✗] Failed to start peripherals! Check robot manually."
        echo "    ssh unitree@${ROBOT_IP} 'cat ~/logs/peripheral_*.log | tail -20'"
    fi
}

# Function to stop peripherals on robot
stop_peripherals() {
    echo ""
    echo "[3/3] Stopping peripherals on robot..."
    ssh -n ${SSH_OPTS} ${ROBOT_USER}@${ROBOT_IP} 'pkill -f robot_peripherals.py 2>/dev/null; exit 0' || true
    echo "[✓] Peripherals stopped"
}

# Cleanup function - called on exit
cleanup() {
    echo ""
    echo "=============================================="
    echo "  Shutting down..."
    echo "=============================================="
    stop_peripherals
    echo ""
    echo "To view peripheral logs:"
    echo "  ssh ${ROBOT_USER}@${ROBOT_IP} 'ls -lt ~/logs/ | head'"
}

# Register cleanup on script exit
trap cleanup EXIT

# Clear stale Redis keys from previous sessions
echo "Clearing stale Redis keys..."
redis-cli DEL teleop_state_info loco_vel_cmd sim2real_ready > /dev/null 2>&1
echo "Redis keys cleared."
echo ""

# Start peripherals
start_peripherals

# Run the RL policy with hybrid locomotion mode
echo ""
echo "[2/3] Starting hybrid locomotion mode..."
echo "=============================================="
cd ${SCRIPT_DIR}/deploy_real

python server_low_level_g1_real.py \
    --policy ${CKPT_PATH} \
    --net ${NET_INTERFACE} \
    --device cuda \
    --hybrid_loco_mode \
    --smooth_body 0.5
