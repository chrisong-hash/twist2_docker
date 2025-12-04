#!/bin/bash

# TWIST2 Docker - Run Script
# Start or enter the TWIST2 container

SCRIPT_DIR=$(dirname $(realpath $0))
PROJECT_ROOT=$(dirname $SCRIPT_DIR)

cd "$PROJECT_ROOT"

# Check if container exists
if docker ps -a --format '{{.Names}}' | grep -q "^twist2$"; then
    # Container exists
    if docker ps --format '{{.Names}}' | grep -q "^twist2$"; then
        # Container is running
        echo "✅ Container 'twist2' is already running"
        echo ""
        echo "To enter the container:"
        echo "  docker exec -it twist2 bash"
        echo ""
        echo "Or run a command directly:"
        echo "  docker exec -it twist2 bash -c 'cd /workspace/twist2 && bash sim2sim.sh'"
    else
        # Container exists but not running
        echo "Starting existing container 'twist2'..."
        docker compose up -d
        echo "✅ Container started"
        echo ""
        echo "To enter the container:"
        echo "  docker exec -it twist2 bash"
    fi
else
    # Container doesn't exist
    echo "Container 'twist2' not found. Creating..."
    
    # Check if image exists
    if ! docker images --format '{{.Repository}}' | grep -q "twist2_ig"; then
        echo "Image not found. Building first..."
        docker compose build
    fi
    
    docker compose up -d
    echo "✅ Container created and started"
    echo ""
    echo "First time? Verify the setup:"
    echo "  docker exec -it twist2 bash"
    echo "  cd /workspace/twist2"
    echo "  bash verify_docker_setup.sh"
    echo ""
    echo "Then test simulation:"
    echo "  bash sim2sim.sh"
fi

echo ""
echo "Quick commands:"
echo "  Enter container:   docker exec -it twist2 bash"
echo "  Stop container:    docker compose down"
echo "  View logs:         docker logs twist2"
echo "  Rebuild:           ./scripts/rebuild_docker.sh"
echo ""

