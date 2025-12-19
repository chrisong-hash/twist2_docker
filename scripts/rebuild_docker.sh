#!/bin/bash

# TWIST2 Docker - Rebuild Script
# Rebuild the Docker image from scratch

set -e

SCRIPT_DIR=$(dirname $(realpath $0))
PROJECT_ROOT=$(dirname $SCRIPT_DIR)

# Enable BuildKit for pip cache mounting (speeds up rebuilds)
export DOCKER_BUILDKIT=1

echo "========================================"
echo "TWIST2 Docker Rebuild"
echo "========================================"
echo ""

cd "$PROJECT_ROOT"

# Ask for confirmation
read -p "This will rebuild the Docker image from scratch. Continue? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "Step 1: Stopping and removing existing container..."
docker compose down --remove-orphans
echo "✅ Container stopped and removed"

echo ""
echo "Step 2: Removing old images (optional cleanup)..."
read -p "Remove old Docker images to save space? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    docker rmi twist2_ig 2>/dev/null || echo "  (No old image to remove)"
    docker system prune -f
    echo "✅ Cleanup complete"
fi

echo ""
echo "Step 3: Building new Docker image..."
echo "This will take ~10-15 minutes (faster on subsequent builds due to pip caching)..."
echo ""

# Use BuildKit for cache mounts (pip downloads are cached between builds)
if DOCKER_BUILDKIT=1 docker compose build; then
    echo ""
    echo "✅ Docker image rebuilt successfully!"
else
    echo ""
    echo "❌ Docker build failed!"
    exit 1
fi

echo ""
echo "Step 4: Starting new container..."
docker compose up -d

echo ""
echo "========================================"
echo "Rebuild Complete!"
echo "========================================"
echo ""
echo "Verify the new setup:"
echo "  docker exec -it twist2 bash"
echo "  cd /workspace/twist2"
echo "  bash verify_docker_setup.sh"
echo ""
echo "If everything passes, test simulation:"
echo "  bash sim2sim.sh"
echo ""

