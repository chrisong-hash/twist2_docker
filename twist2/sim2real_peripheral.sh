#!/bin/bash
# TWIST2 Sim2Real with automatic peripheral management
# Starts robot_peripherals.py on robot, runs RL policy, cleans up on exit

# ============== Configuration ==============
ROBOT_USER="unitree"
ROBOT_IP="192.168.123.164"
PC_IP="192.168.123.222"  # PC IP as seen from robot (for Redis)

# Network interface connecting to robot
NET_INTERFACE="enp4s0"

# SSH options (use identity file, skip host key check for convenience)
SSH_OPTS="-i ~/.ssh/id_robot -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=5"

# ============================================

echo "=============================================="
echo "  TWIST2 Sim2Real + Peripherals"
echo "=============================================="
echo "  Robot:     ${ROBOT_USER}@${ROBOT_IP}"
echo "  PC Redis:  ${PC_IP}"
echo "  Network:   ${NET_INTERFACE}"
echo "=============================================="
echo ""

# Activate conda environment (flexible path for Docker/host)
if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source /opt/conda/etc/profile.d/conda.sh
    conda activate twist2 2>/dev/null || true
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source $HOME/miniconda3/etc/profile.d/conda.sh
    conda activate twist2
fi
# If already in twist2 env (e.g., Docker), continue

SCRIPT_DIR=$(dirname $(realpath $0))
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
    
    # Start peripherals with logging (use ssh -f to fork SSH to background)
    echo "    Launching peripheral script..."
    LOG_FILE="peripheral_\$(date +%Y%m%d_%H%M%S).log"
    ssh -f ${SSH_OPTS} ${ROBOT_USER}@${ROBOT_IP} "cd ~ && python3 robot_peripherals.py --redis ${PC_IP} > ~/logs/${LOG_FILE} 2>&1"
    
    sleep 3
    
    # Verify it started
    echo "    Verifying..."
    if ssh -n ${SSH_OPTS} ${ROBOT_USER}@${ROBOT_IP} 'pgrep -f robot_peripherals.py' > /dev/null 2>&1; then
        echo "[✓] Peripherals started (logging to ~/logs/)"
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

# Cleanup function - called on exit (normal, Ctrl+C, or error)
cleanup() {
    echo ""
    echo "=============================================="
    echo "  Shutting down..."
    echo "=============================================="
    stop_peripherals
    echo ""
    echo "To view peripheral logs:"
    echo "  ssh ${ROBOT_USER}@${ROBOT_IP} 'ls -lt ~/logs/ | head'"
    echo "  ssh ${ROBOT_USER}@${ROBOT_IP} 'cat ~/logs/peripheral_LATEST.log'"
}

# Register cleanup on script exit
trap cleanup EXIT

# Start peripherals
start_peripherals

# Run the RL policy
echo ""
echo "[2/3] Starting RL policy..."
echo "=============================================="
cd ${SCRIPT_DIR}/deploy_real

python server_low_level_g1_real.py \
    --policy ${CKPT_PATH} \
    --net ${NET_INTERFACE} \
    --device cuda \
    --use_hand \
    --smooth_body 0.5
