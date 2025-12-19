#!/bin/bash

# Joystick Teleoperation Script
# Controls G1 locomotion via joystick while maintaining upper body tracking
#
# Architecture:
# 1. Python teleop (xrobot_teleop_inspire.py): Captures Pico controller + upper body tracking -> Redis
# 2. C++ LocoClient bridge (g1_loco_redis_bridge): Reads joystick from Redis -> LocoClient (legs)
# 3. Python policy server (server_low_level_g1_real.py --use_arm_sdk): Reads mimic_obs -> arm_sdk (upper body)

set -e

SCRIPT_DIR="$(dirname "$0")"
WORKSPACE_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║       G1 Joystick Teleoperation (Hybrid Architecture)        ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if we're inside the Docker container
if [ ! -d "/workspace" ]; then
    echo -e "${RED}Error: This script must be run inside the twist2 Docker container${NC}"
    echo "Run: docker exec -it twist2 bash"
    echo "Then: cd /workspace/twist2 && bash joystick_teleop.sh"
    exit 1
fi

# Make sure Redis is running
redis-cli ping > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}Starting Redis server...${NC}"
    redis-server --daemonize yes
    sleep 1
fi
echo -e "${GREEN}✓ Redis is running${NC}"

# Check if loco_bridge binary exists
LOCO_BRIDGE="/workspace/unitree_sdk2/build/bin/g1_loco_redis_bridge"
if [ ! -f "$LOCO_BRIDGE" ]; then
    echo -e "${YELLOW}Building loco_bridge...${NC}"
    cd /workspace/unitree_sdk2/build
    make g1_loco_redis_bridge
    cd "$SCRIPT_DIR"
fi
echo -e "${GREEN}✓ Loco bridge binary ready${NC}"

# Navigate to deployment directory
cd "$SCRIPT_DIR/deploy_real" || exit 1

# Initialize conda (will activate specific environments per component)
eval "$(conda shell.bash hook)"

# Configuration
actual_human_height="${HUMAN_HEIGHT:-1.65}"
redis_ip="localhost"
network_interface="${NETWORK_INTERFACE:-enp4s0}"
policy_path="${POLICY_PATH:-/workspace/twist2/assets/ckpts/twist2_1017_20k.onnx}"
config_path="robot_control/configs/g1.yaml"

# Inspire hand configuration (set USE_INSPIRE_HANDS=1 to enable)
USE_INSPIRE_HANDS="${USE_INSPIRE_HANDS:-0}"
inspire_left_ip="192.168.123.210"
inspire_right_ip="192.168.123.211"

echo ""
echo -e "${CYAN}=== Configuration ===${NC}"
echo "  Network Interface: $network_interface"
echo "  Redis IP: $redis_ip"
echo "  Human Height: $actual_human_height"
echo "  Policy: $policy_path"
echo ""
echo -e "${CYAN}=== Architecture ===${NC}"
echo "  ┌─────────────────┐    ┌─────────────┐    ┌─────────────────┐"
echo "  │   Pico VR       │───▶│   Redis     │───▶│  LocoClient     │"
echo "  │   Controller    │    │   (IPC)     │    │  (Legs)         │"
echo "  └─────────────────┘    └─────────────┘    └─────────────────┘"
echo "                               │"
echo "                               ▼"
echo "                         ┌─────────────┐    ┌─────────────────┐"
echo "                         │  TWIST2     │───▶│  arm_sdk        │"
echo "                         │  Policy     │    │  (Upper Body)   │"
echo "                         └─────────────┘    └─────────────────┘"
echo ""
echo -e "${CYAN}=== Controls ===${NC}"
echo "  Left Joystick:   Forward/Backward & Strafe"
echo "  Right Joystick:  Turn Left/Right"
echo "  B Button (hold): Sprint (2x speed)"
echo "  A Button:        Stop locomotion bridge"
echo "  Triggers:        Open/Close Inspire hands"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down...${NC}"
    
    # Kill processes in reverse order using SIGTERM then SIGKILL
    if [ ! -z "$POLICY_PID" ] && kill -0 $POLICY_PID 2>/dev/null; then
        echo "Stopping Policy server (PID: $POLICY_PID)..."
        kill -TERM $POLICY_PID 2>/dev/null
        sleep 1
        kill -KILL $POLICY_PID 2>/dev/null || true
    fi
    
    if [ ! -z "$LOCO_PID" ] && kill -0 $LOCO_PID 2>/dev/null; then
        echo "Stopping Loco bridge (PID: $LOCO_PID)..."
        kill -TERM $LOCO_PID 2>/dev/null
        sleep 1
        kill -KILL $LOCO_PID 2>/dev/null || true
    fi
    
    if [ ! -z "$TELEOP_PID" ] && kill -0 $TELEOP_PID 2>/dev/null; then
        echo "Stopping Python teleop (PID: $TELEOP_PID)..."
        kill -TERM $TELEOP_PID 2>/dev/null
        sleep 1
        kill -KILL $TELEOP_PID 2>/dev/null || true
    fi
    
    # Also kill any orphaned processes
    pkill -f "xrobot_teleop_inspire" 2>/dev/null || true
    pkill -f "g1_loco_redis_bridge" 2>/dev/null || true
    pkill -f "server_low_level_g1_real" 2>/dev/null || true
    
    echo -e "${GREEN}Shutdown complete.${NC}"
    exit 0
}

# Set up signal handlers - trap on multiple signals
trap cleanup SIGINT SIGTERM EXIT

# ========================================
# Step 1: Start Python teleoperation (gmr environment)
# ========================================
echo -e "${GREEN}[1/3] Starting Python teleoperation (Pico capture)...${NC}"

# Build inspire hand arguments if enabled
INSPIRE_ARGS=""
if [ "$USE_INSPIRE_HANDS" = "1" ]; then
    echo "  Inspire Hands: ENABLED"
    INSPIRE_ARGS="--use_inspire_hands --inspire_left_ip $inspire_left_ip --inspire_right_ip $inspire_right_ip"
else
    echo "  Inspire Hands: DISABLED (set USE_INSPIRE_HANDS=1 to enable)"
fi

# Use gmr environment for teleop (has loop_rate_limiters, mink, etc.)
conda activate gmr
python xrobot_teleop_inspire.py \
    --robot unitree_g1 \
    --actual_human_height $actual_human_height \
    --redis_ip $redis_ip \
    --target_fps 100 \
    --measure_fps 0 \
    $INSPIRE_ARGS &
TELEOP_PID=$!

sleep 3

if ! kill -0 $TELEOP_PID 2>/dev/null; then
    echo -e "${RED}Error: Python teleop failed to start${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python teleop running (PID: $TELEOP_PID)${NC}"

# ========================================
# Step 2: Start C++ LocoClient bridge
# ========================================
echo -e "${GREEN}[2/3] Starting LocoClient bridge (legs locomotion)...${NC}"
$LOCO_BRIDGE --network_interface $network_interface --redis_ip $redis_ip &
LOCO_PID=$!

sleep 2

if ! kill -0 $LOCO_PID 2>/dev/null; then
    echo -e "${RED}Error: Loco bridge failed to start${NC}"
    cleanup
    exit 1
fi
echo -e "${GREEN}✓ Loco bridge running (PID: $LOCO_PID)${NC}"

# ========================================
# Step 3: Start TWIST2 Policy Server (twist2 environment)
# ========================================
# TEMPORARILY DISABLED - Testing LocoClient alone first
echo -e "${YELLOW}[3/3] Policy server DISABLED for testing LocoClient${NC}"
echo -e "${YELLOW}      Robot will only respond to joystick (no upper body tracking)${NC}"
POLICY_PID=""

# TODO: Re-enable after confirming LocoClient works:
# echo -e "${GREEN}[3/3] Starting TWIST2 Policy server (upper body via arm_sdk)...${NC}"
# conda activate twist2
# python server_low_level_g1_real.py \
#     --policy "$policy_path" \
#     --config "$config_path" \
#     --net $network_interface \
#     --use_arm_sdk &
# POLICY_PID=$!
# sleep 3
# if ! kill -0 $POLICY_PID 2>/dev/null; then
#     echo -e "${RED}Error: Policy server failed to start${NC}"
#     cleanup
#     exit 1
# fi
# echo -e "${GREEN}✓ Policy server running (PID: $POLICY_PID)${NC}"

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║              All systems running! Press Ctrl+C to stop       ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Note: Use the Pico VR controller to enter teleop mode (Right key_one)${NC}"
echo -e "${YELLOW}      Then use the joystick to control locomotion${NC}"
echo ""

# Wait for any process to exit
wait -n $TELEOP_PID $LOCO_PID $POLICY_PID

# Cleanup
cleanup
