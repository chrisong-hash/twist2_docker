#!/bin/bash

# Verification script for hybrid teleoperation setup
# Checks that all components are in place and configured correctly

echo "=========================================="
echo "  Hybrid Teleop Setup Verification"
echo "=========================================="
echo ""

SCRIPT_DIR=$(dirname $(realpath $0))
DEPLOY_DIR="${SCRIPT_DIR}/deploy_real"

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✓${NC} $2"
        return 0
    else
        echo -e "${RED}✗${NC} $2 (missing: $1)"
        return 1
    fi
}

check_dir() {
    if [ -d "$1" ]; then
        echo -e "${GREEN}✓${NC} $2"
        return 0
    else
        echo -e "${RED}✗${NC} $2 (missing: $1)"
        return 1
    fi
}

# Check core files
echo "Checking core files..."
check_file "${DEPLOY_DIR}/xrobot_teleop_hybrid.py" "Main hybrid teleop script"
check_file "${DEPLOY_DIR}/hybrid_state_machine.py" "State machine"
check_file "${DEPLOY_DIR}/HYBRID_TELEOP_README.md" "Documentation"
check_file "${SCRIPT_DIR}/teleop_hybrid.sh" "Launch script"
echo ""

# Check Xsens/Manus integration
echo "Checking Xsens/Manus integration..."
check_dir "${DEPLOY_DIR}/xsens_manus_integration" "Integration folder"
check_file "${DEPLOY_DIR}/xsens_manus_integration/__init__.py" "Module init"
check_file "${DEPLOY_DIR}/xsens_manus_integration/xsens_streamer.py" "Xsens streamer"
check_file "${DEPLOY_DIR}/xsens_manus_integration/udp_listener.py" "UDP listener"
check_file "${DEPLOY_DIR}/xsens_manus_integration/inspire_hand_streamer_v2.py" "Inspire mapper"
check_file "${DEPLOY_DIR}/xsens_manus_integration/offsets.json" "Calibration offsets"
echo ""

# Check Python imports (basic syntax check)
echo "Checking Python syntax..."
cd "${DEPLOY_DIR}"
python3 -m py_compile xrobot_teleop_hybrid.py 2>/dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} xrobot_teleop_hybrid.py syntax OK"
else
    echo -e "${RED}✗${NC} xrobot_teleop_hybrid.py has syntax errors"
fi

python3 -m py_compile hybrid_state_machine.py 2>/dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} hybrid_state_machine.py syntax OK"
else
    echo -e "${RED}✗${NC} hybrid_state_machine.py has syntax errors"
fi
echo ""

# Check Redis
echo "Checking Redis..."
redis-cli ping > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} Redis is running"
else
    echo -e "${YELLOW}⚠${NC} Redis is not running (will be started by launch script)"
fi
echo ""

# Check network (if robot IP is reachable)
echo "Checking network..."
ping -c 1 -W 1 192.168.123.210 > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} Left Inspire hand reachable (192.168.123.210)"
else
    echo -e "${YELLOW}⚠${NC} Left Inspire hand not reachable (may not be powered)"
fi

ping -c 1 -W 1 192.168.123.211 > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} Right Inspire hand reachable (192.168.123.211)"
else
    echo -e "${YELLOW}⚠${NC} Right Inspire hand not reachable (may not be powered)"
fi

ping -c 1 -W 1 192.168.123.164 > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} Robot reachable (192.168.123.164)"
else
    echo -e "${YELLOW}⚠${NC} Robot not reachable (may not be powered)"
fi
echo ""

# Check conda environment
echo "Checking conda environment..."
if conda env list | grep -q "gmr"; then
    echo -e "${GREEN}✓${NC} Conda environment 'gmr' exists"
else
    echo -e "${RED}✗${NC} Conda environment 'gmr' not found"
fi
echo ""

# Summary
echo "=========================================="
echo "  Verification Complete"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Start low-level controller: bash sim2real.sh"
echo "2. Start hybrid teleop: bash teleop_hybrid.sh"
echo ""
echo "See HYBRID_TELEOP_README.md for detailed usage instructions."
