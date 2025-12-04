#!/bin/bash

# TWIST2 Docker - Clean All Script
# Remove everything to test fresh installation

SCRIPT_DIR=$(dirname $(realpath $0))
PROJECT_ROOT=$(dirname $SCRIPT_DIR)

cd "$PROJECT_ROOT"

echo "========================================"
echo "TWIST2 Docker - Clean All"
echo "========================================"
echo ""
echo "This will remove:"
echo "  - twist2 container"
echo "  - twist2_ig Docker image"
echo "  - Docker build cache"
echo ""

read -p "Are you sure you want to continue? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "Step 1: Stopping and removing container..."
docker compose down --remove-orphans || true
echo "✅ Container removed"

echo ""
echo "Step 2: Removing Docker image..."
docker rmi twist2_ig 2>/dev/null || echo "  (No image to remove)"
echo "✅ Image removed"

echo ""
echo "Step 3: Cleaning Docker cache..."
docker system prune -f
echo "✅ Cache cleaned"

echo ""
echo "Step 4: Verifying cleanup..."
if docker ps -a --format '{{.Names}}' | grep -q "^twist2$"; then
    echo "⚠️  Container still exists"
else
    echo "✅ No container found"
fi

if docker images --format '{{.Repository}}' | grep -q "twist2_ig"; then
    echo "⚠️  Image still exists"
else
    echo "✅ No image found"
fi

echo ""
echo "========================================"
echo "Cleanup Complete!"
echo "========================================"
echo ""
echo "You can now test fresh installation:"
echo "  ./scripts/install.sh"
echo ""

