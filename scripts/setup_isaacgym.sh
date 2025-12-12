#!/bin/bash

# TWIST2 Docker - IsaacGym Setup Script
# Automatically downloads and sets up IsaacGym

set -e

SCRIPT_DIR=$(dirname $(realpath $0))
PROJECT_ROOT=$(dirname $SCRIPT_DIR)
ISAACGYM_DIR="$PROJECT_ROOT/isaacgym"

echo "========================================"
echo "IsaacGym Setup for TWIST2"
echo "========================================"
echo ""

# Check if isaacgym already exists
if [ -d "$ISAACGYM_DIR" ]; then
    echo "✅ IsaacGym directory already exists at: $ISAACGYM_DIR"
    
    # Verify it has the expected structure
    if [ -f "$ISAACGYM_DIR/python/setup.py" ]; then
        echo "✅ IsaacGym structure looks valid"
        exit 0
    else
        echo "⚠️  IsaacGym directory exists but structure looks incomplete"
        read -p "   Remove and re-download? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$ISAACGYM_DIR"
        else
            exit 1
        fi
    fi
fi

echo "📥 IsaacGym not found. Setting up..."
echo ""

# Create temporary directory
TMP_DIR=$(mktemp -d)
trap "rm -rf $TMP_DIR" EXIT

cd "$TMP_DIR"

# Check if user provided path to downloaded archive
if [ -n "$1" ] && [ -f "$1" ]; then
    echo "✅ Using provided archive: $1"
    cp "$1" isaacgym.tar.gz
elif [ -f "$PROJECT_ROOT/IsaacGym_Preview_4_Package.tar.gz" ]; then
    echo "✅ Found IsaacGym archive in project root"
    cp "$PROJECT_ROOT/IsaacGym_Preview_4_Package.tar.gz" isaacgym.tar.gz
elif [ -f "$HOME/Downloads/IsaacGym_Preview_4_Package.tar.gz" ]; then
    echo "✅ Found IsaacGym archive in ~/Downloads"
    cp "$HOME/Downloads/IsaacGym_Preview_4_Package.tar.gz" isaacgym.tar.gz
else
    echo "Attempting to download IsaacGym Preview 4..."
    echo ""

    # Try to download IsaacGym
    # Note: NVIDIA may require login/registration, so we try multiple methods

    # Method 1: Direct download (may work without authentication)
    ISAACGYM_URL="https://developer.nvidia.com/isaac-gym-preview-4-download"
    ISAACGYM_DIRECT="https://developer.nvidia.com/downloads/isaac-gym-preview-4"

    echo "Method 1: Attempting direct download..."
    if wget --quiet --show-progress -O isaacgym.tar.gz "$ISAACGYM_DIRECT" 2>/dev/null; then
        echo "✅ Download successful!"
    elif wget --quiet --show-progress -O isaacgym.tar.gz "https://developer.download.nvidia.com/isaac-gym/isaac-gym-preview-4.tar.gz" 2>/dev/null; then
        echo "✅ Download successful!"
    else
        echo ""
        echo "❌ Automatic download failed (NVIDIA may require authentication)"
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "MANUAL DOWNLOAD REQUIRED"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        echo "Please download IsaacGym manually:"
        echo ""
        echo "1. Visit: https://developer.nvidia.com/isaac-gym"
        echo "2. Click 'Download' (you may need to register/login)"
        echo "3. Download: Isaac Gym Preview 4"
        echo "4. Save the file (IsaacGym_Preview_4_Package.tar.gz)"
        echo ""
        echo "Then run one of these commands:"
        echo ""
        echo "  Option A - Save to project root and extract:"
        echo "    cd $PROJECT_ROOT"
        echo "    tar -xzf IsaacGym_Preview_4_Package.tar.gz"
        echo ""
        echo "  Option B - Point this script to your download:"
        echo "    bash $0 /path/to/IsaacGym_Preview_4_Package.tar.gz"
        echo ""
        echo "  Option C - Move to ~/Downloads and re-run:"
        echo "    mv /path/to/IsaacGym_Preview_4_Package.tar.gz ~/Downloads/"
        echo "    bash $0"
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        exit 1
    fi
fi

# Extract the archive
echo ""
echo "📦 Extracting IsaacGym..."
tar -xzf isaacgym.tar.gz

# Move to project root
if [ -d "isaacgym" ]; then
    mv isaacgym "$ISAACGYM_DIR"
    echo "✅ IsaacGym extracted to: $ISAACGYM_DIR"
elif [ -d "IsaacGym_Preview_4_Package/isaacgym" ]; then
    mv IsaacGym_Preview_4_Package/isaacgym "$ISAACGYM_DIR"
    echo "✅ IsaacGym extracted to: $ISAACGYM_DIR"
else
    echo "❌ Failed to find isaacgym directory in archive"
    echo "   Please extract manually"
    exit 1
fi

# Verify installation
if [ -f "$ISAACGYM_DIR/python/setup.py" ]; then
    echo "✅ IsaacGym setup complete!"
    echo ""
    echo "Next step: Run ./scripts/install.sh to build Docker image"
else
    echo "❌ IsaacGym structure verification failed"
    echo "   Expected: $ISAACGYM_DIR/python/setup.py"
    exit 1
fi

echo ""

