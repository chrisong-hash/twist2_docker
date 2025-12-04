#!/bin/bash

# Quick automated test for TWIST2 capabilities
# This script performs basic checks without requiring multiple terminals

set -e

SCRIPT_DIR=$(dirname $(realpath $0))
cd "$SCRIPT_DIR"

echo "========================================"
echo "TWIST2 Quick Test Suite"
echo "========================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

function test_pass() {
    echo -e "${GREEN}✓ PASS${NC}: $1"
    TESTS_PASSED=$((TESTS_PASSED + 1))
}

function test_fail() {
    echo -e "${RED}✗ FAIL${NC}: $1"
    TESTS_FAILED=$((TESTS_FAILED + 1))
}

function test_warn() {
    echo -e "${YELLOW}⚠ WARN${NC}: $1"
}

echo "TEST 1: Environment Setup"
echo "----------------------------"

# Check Python
if python -c "import isaacgym" 2>/dev/null; then
    test_pass "IsaacGym installed"
else
    test_fail "IsaacGym not found"
fi

if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    test_pass "PyTorch with CUDA available"
else
    test_fail "PyTorch CUDA not available"
fi

if python -c "import redis" 2>/dev/null; then
    test_pass "Redis Python module installed"
else
    test_fail "Redis Python module not found"
fi

if python -c "import onnxruntime" 2>/dev/null; then
    test_pass "ONNXRuntime installed"
else
    test_fail "ONNXRuntime not found"
fi

echo ""
echo "TEST 2: Files and Assets"
echo "----------------------------"

if [ -f "assets/ckpts/twist2_1017_20k.onnx" ]; then
    test_pass "Pretrained checkpoint exists"
else
    test_fail "Pretrained checkpoint not found"
fi

motion_count=$(ls -1 assets/example_motions/*.pkl 2>/dev/null | wc -l || echo 0)
if [ "$motion_count" -gt 0 ]; then
    test_pass "Found $motion_count example motion files"
else
    test_fail "No example motion files found"
fi

if [ -d "../isaacgym" ]; then
    test_pass "Isaac Gym directory found"
else
    test_fail "Isaac Gym directory not found at ../isaacgym"
fi

echo ""
echo "TEST 3: Redis Server"
echo "----------------------------"

# Check if Redis is running
if pgrep -x "redis-server" > /dev/null; then
    test_pass "Redis server is running"
else
    test_warn "Redis server not running, attempting to start..."
    redis-server --daemonize yes
    sleep 2
    if pgrep -x "redis-server" > /dev/null; then
        test_pass "Redis server started successfully"
    else
        test_fail "Failed to start Redis server"
    fi
fi

# Test Redis connection
if python -c "import redis; r = redis.Redis(host='localhost', port=6379); r.ping()" 2>/dev/null; then
    test_pass "Redis connection successful"
else
    test_fail "Cannot connect to Redis"
fi

echo ""
echo "TEST 4: Import Check"
echo "----------------------------"

# Check if key modules can be imported
if python -c "
import sys
sys.path.insert(0, 'legged_gym')
import legged_gym
print('legged_gym imports successfully')
" 2>/dev/null; then
    test_pass "legged_gym module imports correctly"
else
    test_fail "legged_gym module import failed"
fi

if python -c "
import sys
sys.path.insert(0, 'rsl_rl')
import rsl_rl
print('rsl_rl imports successfully')
" 2>/dev/null; then
    test_pass "rsl_rl module imports correctly"
else
    test_fail "rsl_rl module import failed"
fi

echo ""
echo "TEST 5: Script Files"
echo "----------------------------"

for script in train.sh sim2sim.sh sim2real.sh eval.sh run_motion_server.sh gui.sh; do
    if [ -f "$script" ] && [ -x "$script" ]; then
        test_pass "Script $script exists and is executable"
    elif [ -f "$script" ]; then
        test_warn "Script $script exists but not executable (chmod +x $script)"
        chmod +x "$script"
    else
        test_fail "Script $script not found"
    fi
done

echo ""
echo "========================================"
echo "TEST SUMMARY"
echo "========================================"
echo -e "${GREEN}Tests Passed: $TESTS_PASSED${NC}"
if [ $TESTS_FAILED -gt 0 ]; then
    echo -e "${RED}Tests Failed: $TESTS_FAILED${NC}"
else
    echo -e "${GREEN}Tests Failed: 0${NC}"
fi
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed! Your environment is ready.${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Test simulation:  bash sim2sim.sh  (in one terminal)"
    echo "                     bash run_motion_server.sh  (in another terminal)"
    echo "2. Test training:    bash train.sh test_run cuda:0"
    echo "3. Use GUI:          bash gui.sh"
    echo ""
    echo "See VERIFICATION_GUIDE.md for detailed instructions."
else
    echo -e "${RED}✗ Some tests failed. Please fix the issues above.${NC}"
    exit 1
fi

