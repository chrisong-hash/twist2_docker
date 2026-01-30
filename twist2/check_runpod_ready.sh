#!/bin/bash
# Check if RunPod environment is ready for V7 training
# Run this script on RunPod to verify all dependencies and data

echo "=========================================="
echo "RunPod V7 Training Readiness Check"
echo "=========================================="

ERRORS=0
WARNINGS=0

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

check_pass() {
    echo -e "${GREEN}✓${NC} $1"
}

check_fail() {
    echo -e "${RED}✗${NC} $1"
    ((ERRORS++))
}

check_warn() {
    echo -e "${YELLOW}!${NC} $1"
    ((WARNINGS++))
}

echo ""
echo "=== 1. Python Environment ==="

# Check conda
if command -v conda &> /dev/null; then
    check_pass "Conda installed"
    CONDA_ENV=$(conda info --envs | grep '*' | awk '{print $1}')
    echo "    Active env: $CONDA_ENV"
else
    check_warn "Conda not found (using system Python)"
fi

# Check Python version
PYTHON_VERSION=$(python --version 2>&1)
if [[ $PYTHON_VERSION == *"3.8"* ]]; then
    check_pass "Python 3.8.x ($PYTHON_VERSION)"
else
    check_warn "Python version: $PYTHON_VERSION (3.8 recommended for Isaac Gym)"
fi

echo ""
echo "=== 2. GPU ==="

# Check NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
    check_pass "GPU: $GPU_NAME ($GPU_MEM)"
else
    check_fail "nvidia-smi not found - no GPU?"
fi

# Check CUDA
if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null)
    check_pass "PyTorch CUDA: $CUDA_VERSION"
else
    check_fail "PyTorch CUDA not available"
fi

echo ""
echo "=== 3. Required Python Packages ==="

check_python_package() {
    if python -c "import $1" 2>/dev/null; then
        check_pass "$1"
    else
        check_fail "$1 not installed"
    fi
}

check_python_package "torch"
check_python_package "isaacgym"
check_python_package "numpy"
check_python_package "wandb"
check_python_package "rich"
check_python_package "cv2"
check_python_package "yaml"
check_python_package "matplotlib"

echo ""
echo "=== 4. Code Repository ==="

# Check if twist2 directory exists
TWIST2_DIR="/workspace/twist2_docker/twist2"
if [ -d "$TWIST2_DIR" ]; then
    check_pass "twist2 directory exists"
else
    TWIST2_DIR="/workspace/twist2"
    if [ -d "$TWIST2_DIR" ]; then
        check_pass "twist2 directory exists (alt path)"
    else
        check_fail "twist2 directory not found"
        TWIST2_DIR=""
    fi
fi

if [ -n "$TWIST2_DIR" ]; then
    # Check key files
    if [ -f "$TWIST2_DIR/legged_gym/legged_gym/envs/g1/g1_mimic_distill_config_v7.py" ]; then
        check_pass "V7 config exists"
    else
        check_fail "V7 config not found"
    fi
    
    if [ -f "$TWIST2_DIR/legged_gym/legged_gym/envs/g1/g1_mimic_distill_freeze.py" ]; then
        check_pass "V7 freeze env exists"
    else
        check_fail "V7 freeze env not found"
    fi
fi

echo ""
echo "=== 5. Motion Data ==="

# Check motion data directories
MOTION_DATA_DIR="$TWIST2_DIR/legged_gym/resources/motions"

check_motion_dir() {
    local dir_name=$1
    local min_files=${2:-1}
    local full_path="$MOTION_DATA_DIR/$dir_name"
    
    if [ -d "$full_path" ]; then
        file_count=$(find "$full_path" -name "*.pkl" 2>/dev/null | wc -l)
        if [ "$file_count" -ge "$min_files" ]; then
            check_pass "$dir_name: $file_count .pkl files"
        else
            check_warn "$dir_name: only $file_count .pkl files (expected $min_files+)"
        fi
    else
        check_fail "$dir_name directory not found"
    fi
}

if [ -d "$MOTION_DATA_DIR" ]; then
    check_pass "Motion data directory exists"
    
    # Check main motion directories
    check_motion_dir "TWIST2_full" 100
    check_motion_dir "virtuals" 10
    check_motion_dir "c_walk" 5
    check_motion_dir "example_motions" 1
else
    check_fail "Motion data directory not found: $MOTION_DATA_DIR"
fi

echo ""
echo "=== 6. Motion Config Files ==="

MOTION_CONFIG_DIR="$TWIST2_DIR/legged_gym/motion_data_configs"

if [ -d "$MOTION_CONFIG_DIR" ]; then
    check_pass "Motion config directory exists"
    
    if [ -f "$MOTION_CONFIG_DIR/walking_focused.yaml" ]; then
        check_pass "walking_focused.yaml exists"
    else
        check_warn "walking_focused.yaml not found (V7 default)"
    fi
    
    if [ -f "$MOTION_CONFIG_DIR/walking_focused_v6_1.yaml" ]; then
        check_pass "walking_focused_v6_1.yaml exists"
    else
        check_warn "walking_focused_v6_1.yaml not found"
    fi
else
    check_fail "Motion config directory not found"
fi

echo ""
echo "=== 7. Assets (URDF) ==="

ASSETS_DIR="$TWIST2_DIR/../assets/g1"
if [ -d "$ASSETS_DIR" ]; then
    check_pass "Assets directory exists"
    
    if [ -f "$ASSETS_DIR/g1_custom_collision_29dof.urdf" ]; then
        check_pass "G1 URDF exists"
    else
        check_fail "G1 URDF not found"
    fi
else
    check_fail "Assets directory not found: $ASSETS_DIR"
fi

echo ""
echo "=== 8. Disk Space ==="

# Check available disk space
AVAIL_SPACE=$(df -h /workspace 2>/dev/null | awk 'NR==2 {print $4}')
USED_PERCENT=$(df -h /workspace 2>/dev/null | awk 'NR==2 {print $5}')

if [ -n "$AVAIL_SPACE" ]; then
    echo "    /workspace: $AVAIL_SPACE available ($USED_PERCENT used)"
    
    # Warn if less than 10GB
    AVAIL_GB=$(df -BG /workspace 2>/dev/null | awk 'NR==2 {print $4}' | tr -d 'G')
    if [ "$AVAIL_GB" -lt 10 ]; then
        check_warn "Low disk space! (<10GB available)"
    else
        check_pass "Sufficient disk space"
    fi
else
    check_warn "Could not check disk space"
fi

echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}All checks passed! Ready for V7 training.${NC}"
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}$WARNINGS warning(s), but should be OK to train.${NC}"
else
    echo -e "${RED}$ERRORS error(s) found. Please fix before training.${NC}"
fi

echo ""
echo "To start V7 training:"
echo "  cd $TWIST2_DIR/legged_gym"
echo "  python legged_gym/scripts/train.py --task g1_priv_mimic_v7 --run_name v7_freeze --proj_name h1 --headless"
echo ""


