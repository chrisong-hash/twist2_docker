#!/bin/bash
# Robot Network Setup Script
# Run this on the robot to:
#   1. Connect to internet via PC
#   2. Start WiFi hotspot for Pico VR
# Usage: sudo bash robot_network_setup.sh

set -e

echo "=========================================="
echo "Robot Network Setup"
echo "=========================================="

# Configuration - adjust if needed
PC_IP="192.168.123.222"      # Your PC's IP on the robot network
DNS_SERVER="8.8.8.8"         # Google DNS
HOTSPOT_NAME="G1-Robot"      # WiFi hotspot name (SSID)
HOTSPOT_PASSWORD="12345678"  # WiFi password (min 8 chars)

echo "PC Gateway IP: $PC_IP"
echo "DNS Server: $DNS_SERVER"
echo "Hotspot SSID: $HOTSPOT_NAME"
echo ""

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root (use sudo)"
   exit 1
fi

# ==========================================
# PART 1: Internet via PC
# ==========================================
echo "=========================================="
echo "PART 1: Setting up internet via PC"
echo "=========================================="

# 1. Delete existing default route
echo "[1/3] Configuring default route..."
ip route del default 2>/dev/null || true
ip route add default via $PC_IP
echo "  Default route set via $PC_IP"

# 2. Set DNS
echo "[2/3] Configuring DNS..."
echo "nameserver $DNS_SERVER" > /etc/resolv.conf
echo "  DNS set to $DNS_SERVER"

# 3. Test connectivity
echo "[3/3] Testing connectivity..."
echo ""

echo "Testing gateway ($PC_IP)..."
if ping -c 1 -W 2 $PC_IP > /dev/null 2>&1; then
    echo "  ✓ Gateway reachable"
else
    echo "  ✗ Gateway not reachable - check PC connection"
    echo "  Continuing anyway (hotspot doesn't need PC)..."
fi

echo "Testing internet (8.8.8.8)..."
if ping -c 2 -W 3 8.8.8.8 > /dev/null 2>&1; then
    echo "  ✓ Internet reachable"
else
    echo "  ✗ Internet not reachable - check PC forwarding setup"
    echo "  Continuing anyway..."
fi

echo "Testing DNS (google.com)..."
if ping -c 2 -W 3 google.com > /dev/null 2>&1; then
    echo "  ✓ DNS working"
else
    echo "  ✗ DNS not working"
    echo "  Continuing anyway..."
fi

# ==========================================
# PART 2: WiFi Hotspot for Pico
# ==========================================
echo ""
echo "=========================================="
echo "PART 2: Setting up WiFi hotspot for Pico"
echo "=========================================="

# 1. Unblock WiFi
echo "[1/4] Enabling WiFi radio..."
nmcli radio wifi on 2>/dev/null || true
rfkill unblock wifi 2>/dev/null || true
sleep 1

# 2. Bring up wlan0
echo "[2/4] Bringing up wlan0..."
ifconfig wlan0 up 2>/dev/null || true
sleep 1

# 3. Check if hotspot already running
echo "[3/4] Checking existing hotspot..."
if nmcli connection show --active | grep -q "$HOTSPOT_NAME"; then
    echo "  Hotspot '$HOTSPOT_NAME' already running"
else
    # Delete old hotspot config if exists
    nmcli connection delete "$HOTSPOT_NAME" 2>/dev/null || true
    
    # Create new hotspot
    echo "[4/4] Creating hotspot '$HOTSPOT_NAME'..."
    nmcli device wifi hotspot ifname wlan0 ssid "$HOTSPOT_NAME" password "$HOTSPOT_PASSWORD"
    sleep 2
fi

# Get hotspot IP
HOTSPOT_IP=$(ip addr show wlan0 | grep "inet " | awk '{print $2}' | cut -d/ -f1)
echo ""
echo "  Hotspot IP: $HOTSPOT_IP"

# ==========================================
# Summary
# ==========================================
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Network status:"
nmcli device status
echo ""
echo "=========================================="
echo "For Pico XRobo App:"
echo "  WiFi Network: $HOTSPOT_NAME"
echo "  WiFi Password: $HOTSPOT_PASSWORD"
echo "  Server IP:     $HOTSPOT_IP"
echo "=========================================="

