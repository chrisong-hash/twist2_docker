#!/bin/bash

echo "========================================"
echo "TWIST2 Docker Setup Verification Script"
echo "========================================"
echo ""

# Check if we're in a Docker container
if [ ! -f /.dockerenv ]; then
    echo "⚠️  WARNING: This script should be run INSIDE the Docker container!"
    echo ""
    echo "To run this script properly:"
    echo "  1. docker exec -it twist2 bash"
    echo "  2. cd /workspace/twist2"
    echo "  3. bash verify_docker_setup.sh"
    echo ""
    exit 1
fi

echo "✓ Running inside Docker container"
echo ""

# Check if we're in the right directory
if [ ! -f "gui.sh" ]; then
    echo "❌ Error: Please run this script from /workspace/twist2"
    echo "   cd /workspace/twist2"
    exit 1
fi

echo "1. Checking Python environment..."
echo "   Python version: $(python --version 2>&1)"
echo "   Python path: $(which python)"
echo ""

echo "2. Checking Python packages..."
python -c "import isaacgym; print('✓ IsaacGym')" 2>/dev/null || echo "❌ IsaacGym not found"
python -c "import torch; print('✓ PyTorch')" 2>/dev/null || echo "❌ PyTorch not found"
python -c "import torch; print(f'  - CUDA available: {torch.cuda.is_available()}'); print(f'  - CUDA devices: {torch.cuda.device_count()}')" 2>/dev/null || echo "❌ PyTorch CUDA check failed"
python -c "import redis; print('✓ Redis')" 2>/dev/null || echo "⚠️  Redis not found (install with: pip install redis)"
python -c "import onnxruntime; print('✓ ONNXRuntime')" 2>/dev/null || echo "⚠️  ONNXRuntime not found (install with: pip install onnxruntime-gpu)"
python -c "import cv2; print('✓ OpenCV')" 2>/dev/null || echo "⚠️  OpenCV not found (install with: pip install opencv-python)"
python -c "import numpy; print('✓ NumPy')" 2>/dev/null || echo "❌ NumPy not found"

echo ""
echo "3. Checking IsaacGym installation..."
if [ -d "/opt/isaacgym" ]; then
    echo "✓ IsaacGym directory exists at /opt/isaacgym"
    echo "  Testing IsaacGym import..."
    python -c "from isaacgym import gymapi; print('✓ IsaacGym gymapi imports successfully')" 2>/dev/null || echo "❌ IsaacGym import failed"
else
    echo "❌ IsaacGym directory not found at /opt/isaacgym"
fi

echo ""
echo "4. Checking GPU/CUDA setup..."
python -c "
import torch
if torch.cuda.is_available():
    print(f'✓ CUDA is available')
    print(f'  - Device count: {torch.cuda.device_count()}')
    print(f'  - Device name: {torch.cuda.get_device_name(0)}')
    print(f'  - CUDA version: {torch.version.cuda}')
else:
    print('❌ CUDA is not available')
" 2>/dev/null || echo "❌ CUDA check failed"

echo ""
echo "5. Checking Redis server..."
if command -v redis-server &> /dev/null; then
    echo "✓ Redis server binary found"
    if pgrep -x "redis-server" > /dev/null; then
        echo "✓ Redis server is running"
    else
        echo "⚠️  Redis server is not running"
        echo "   Start it with: redis-server --daemonize yes"
    fi
else
    echo "❌ Redis server not found (install with: apt-get install redis-server)"
fi

echo ""
echo "6. Checking pretrained checkpoint..."
if [ -f "assets/ckpts/twist2_1017_20k.onnx" ]; then
    echo "✓ Pretrained checkpoint found"
    ls -lh assets/ckpts/twist2_1017_20k.onnx
else
    echo "❌ Pretrained checkpoint not found at assets/ckpts/twist2_1017_20k.onnx"
fi

echo ""
echo "7. Checking example motions..."
if [ -d "assets/example_motions" ] && [ "$(ls -A assets/example_motions/*.pkl 2>/dev/null)" ]; then
    motion_count=$(ls -1 assets/example_motions/*.pkl 2>/dev/null | wc -l)
    echo "✓ Found ${motion_count} example motion files"
    ls assets/example_motions/ | head -5
else
    echo "❌ No example motion files found"
fi

echo ""
echo "8. Checking TWIST2 modules..."
cd /workspace/twist2

# Check legged_gym
if python -c "import sys; sys.path.insert(0, 'legged_gym'); import legged_gym" 2>/dev/null; then
    echo "✓ legged_gym module imports"
else
    echo "⚠️  legged_gym module import failed (may need: cd legged_gym && pip install -e .)"
fi

# Check rsl_rl
if python -c "import sys; sys.path.insert(0, 'rsl_rl'); import rsl_rl" 2>/dev/null; then
    echo "✓ rsl_rl module imports"
else
    echo "⚠️  rsl_rl module import failed (may need: cd rsl_rl && pip install -e .)"
fi

# Check pose
if python -c "import sys; sys.path.insert(0, 'pose'); import pose" 2>/dev/null; then
    echo "✓ pose module imports"
else
    echo "⚠️  pose module import failed (may need: cd pose && pip install -e .)"
fi

echo ""
echo "9. Checking display/X11 setup..."
if [ -n "$DISPLAY" ]; then
    echo "✓ DISPLAY variable set: $DISPLAY"
    if xdpyinfo &>/dev/null; then
        echo "✓ X11 connection working"
    else
        echo "⚠️  Cannot connect to X11 display"
        echo "   On host, run: xhost +local:docker"
    fi
else
    echo "❌ DISPLAY variable not set"
fi

echo ""
echo "10. Checking script files..."
for script in train.sh sim2sim.sh sim2real.sh eval.sh run_motion_server.sh gui.sh; do
    if [ -f "$script" ]; then
        echo "✓ $script exists"
    else
        echo "❌ $script not found"
    fi
done

echo ""
echo "========================================"
echo "Verification Complete!"
echo "========================================"
echo ""
echo "Next steps to test:"
echo "  1. Install missing packages if any"
echo "  2. Start Redis: redis-server --daemonize yes"
echo "  3. Test simulation: bash sim2sim.sh"
echo ""

