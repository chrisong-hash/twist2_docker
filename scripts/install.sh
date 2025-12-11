#!/bin/bash

# TWIST2 Docker - Installation Script
# Run this once to set up everything

set -e

SCRIPT_DIR=$(dirname $(realpath $0))
PROJECT_ROOT=$(dirname $SCRIPT_DIR)

echo "========================================"
echo "TWIST2 Docker Installation"
echo "========================================"
echo ""

cd "$PROJECT_ROOT"

# Initialize git submodules (unitree_sdk2 and GMR)
echo "Step 1: Initializing git submodules..."
if git submodule update --init --recursive; then
    echo "✅ Submodules initialized (unitree_sdk2, GMR)"
else
    echo "⚠️  Warning: Could not initialize submodules (may need to run manually)"
fi
echo ""

# Check if IsaacGym is present
echo "Step 2: Checking IsaacGym..."
if [ -d "isaacgym" ] && [ -d "isaacgym/python" ]; then
    echo "✅ IsaacGym directory found"
else
    echo "❌ IsaacGym directory not found!"
    echo ""
    echo "IsaacGym must be downloaded separately from NVIDIA:"
    echo "  1. Go to: https://developer.nvidia.com/isaac-gym"
    echo "  2. Register and download Isaac Gym Preview 4"
    echo "  3. Extract to: $PROJECT_ROOT/isaacgym"
    echo ""
    echo "Expected structure:"
    echo "  $PROJECT_ROOT/isaacgym/python/"
    echo "  $PROJECT_ROOT/isaacgym/docs/"
    echo ""
    read -p "Continue without IsaacGym? (Docker will fail to build) (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed!"
    echo "   Please install Docker first: https://docs.docker.com/engine/install/"
    exit 1
fi

# Check if docker compose is available
if ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose is not available!"
    echo "   Please install Docker Compose: https://docs.docker.com/compose/install/"
    exit 1
fi

echo "✅ Docker is installed"
echo ""

# Check NVIDIA Docker support
if ! docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi &> /dev/null; then
    echo "⚠️  NVIDIA Docker support not working!"
    echo "   Please install nvidia-docker2:"
    echo "   https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    read -p "   Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✅ NVIDIA Docker support detected"
fi
echo ""

# Set up X11 access
echo "Setting up X11 access for Docker..."
xhost +local:docker >/dev/null 2>&1

# Ask if user wants to make it persistent
read -p "Make X11 access persistent (add to ~/.bashrc)? (Y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    if ! grep -q "xhost +local:docker" ~/.bashrc; then
        echo "# Allow Docker to access X11 display" >> ~/.bashrc
        echo "xhost +local:docker >/dev/null 2>&1" >> ~/.bashrc
        echo "✅ Added to ~/.bashrc"
    else
        echo "✅ Already in ~/.bashrc"
    fi
fi
echo ""

# Build the Docker image
echo "Building Docker image..."
echo "This will take ~10-15 minutes on first run..."
echo ""

cd "$PROJECT_ROOT"

if docker compose build; then
    echo ""
    echo "✅ Docker image built successfully!"
else
    echo ""
    echo "❌ Docker build failed!"
    exit 1
fi

echo ""
echo "========================================"
echo "Installation Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Start container:  ./scripts/run.sh"
echo "  2. Enter container:  docker exec -it twist2 bash"
echo "  3. Verify setup:     cd /workspace/twist2 && bash verify_docker_setup.sh"
echo "  4. Test simulation:  bash sim2sim.sh"
echo ""
echo "See README.md for more information."
echo ""

