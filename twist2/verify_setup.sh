#!/bin/bash

echo "========================================"
echo "TWIST2 Setup Verification Script"
echo "========================================"
echo ""

# Check if we're in the right directory
if [ ! -f "gui.sh" ]; then
    echo "❌ Error: Please run this script from the twist2 directory"
    exit 1
fi

echo "1. Checking Python packages..."
python -c "import isaacgym" && echo "✓ IsaacGym" || echo "❌ IsaacGym not found"
python -c "import torch" && echo "✓ PyTorch" || echo "❌ PyTorch not found"
python -c "import redis" && echo "✓ Redis" || echo "❌ Redis not found"
python -c "import onnxruntime" && echo "✓ ONNXRuntime" || echo "❌ ONNXRuntime not found"
python -c "import cv2" && echo "✓ OpenCV" || echo "❌ OpenCV not found"

echo ""
echo "2. Checking if Redis server is running..."
if pgrep -x "redis-server" > /dev/null; then
    echo "✓ Redis server is running"
else
    echo "⚠ Redis server is not running. Starting it..."
    redis-server --daemonize yes
    sleep 2
    if pgrep -x "redis-server" > /dev/null; then
        echo "✓ Redis server started successfully"
    else
        echo "❌ Failed to start Redis server"
    fi
fi

echo ""
echo "3. Checking GPU availability..."
python -c "import torch; print('✓ CUDA available' if torch.cuda.is_available() else '❌ CUDA not available')"
python -c "import torch; print(f'  GPU Count: {torch.cuda.device_count()}') if torch.cuda.is_available() else None"

echo ""
echo "4. Checking pretrained checkpoint..."
if [ -f "assets/ckpts/twist2_1017_20k.onnx" ]; then
    echo "✓ Pretrained checkpoint found"
else
    echo "❌ Pretrained checkpoint not found"
fi

echo ""
echo "5. Checking example motions..."
if [ -d "assets/example_motions" ] && [ "$(ls -A assets/example_motions/*.pkl 2>/dev/null)" ]; then
    motion_count=$(ls -1 assets/example_motions/*.pkl 2>/dev/null | wc -l)
    echo "✓ Found ${motion_count} example motion files"
else
    echo "❌ No example motion files found"
fi

echo ""
echo "6. Checking Isaac Gym assets..."
if [ -d "../isaacgym" ]; then
    echo "✓ Isaac Gym directory found"
else
    echo "❌ Isaac Gym directory not found"
fi

echo ""
echo "========================================"
echo "Verification Complete!"
echo "========================================"

