#!/bin/bash
# PC Port Forwarding Setup Script
# Run this on your PC to share internet with the robot
# Usage: sudo bash pc_forward_setup.sh

set -e

echo "=========================================="
echo "PC Internet Forwarding Setup"
echo "=========================================="

# Configuration - adjust these if your interfaces are different
INTERNET_IFACE="wlo1"        # Your PC's internet interface (WiFi)
ROBOT_IFACE="enp4s0"         # Your PC's interface to robot network (Ethernet)

echo "Internet interface: $INTERNET_IFACE"
echo "Robot interface: $ROBOT_IFACE"
echo ""

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root (use sudo)"
   exit 1
fi

# 1. Enable IP forwarding
echo "[1/4] Enabling IP forwarding..."
sysctl -w net.ipv4.ip_forward=1
# Make it persistent (optional)
# echo "net.ipv4.ip_forward=1" >> /etc/sysctl.conf

# 2. Set up NAT (MASQUERADE)
echo "[2/4] Setting up NAT..."
# Check if rule already exists
if ! iptables -t nat -C POSTROUTING -o $INTERNET_IFACE -j MASQUERADE 2>/dev/null; then
    iptables -t nat -A POSTROUTING -o $INTERNET_IFACE -j MASQUERADE
    echo "  Added MASQUERADE rule"
else
    echo "  MASQUERADE rule already exists"
fi

# 3. Set up FORWARD rules
echo "[3/4] Setting up FORWARD rules..."
# Add rules at the beginning to bypass Docker rules
iptables -I FORWARD 1 -i $ROBOT_IFACE -o $INTERNET_IFACE -j ACCEPT 2>/dev/null || true
iptables -I FORWARD 2 -i $INTERNET_IFACE -o $ROBOT_IFACE -j ACCEPT 2>/dev/null || true

# 4. Remove any bad routes that might exist
echo "[4/4] Cleaning up bad routes..."
ip route del 8.8.8.8 dev $ROBOT_IFACE 2>/dev/null || true
ip route del 192.168.123.0/24 via 8.8.8.8 2>/dev/null || true

echo ""
echo "=========================================="
echo "Setup complete!"
echo ""
echo "Verification:"
echo "  IP forwarding: $(cat /proc/sys/net/ipv4/ip_forward)"
echo ""
echo "NAT rules:"
iptables -t nat -L POSTROUTING -n -v | grep -E "MASQUERADE|$INTERNET_IFACE" | head -5
echo ""
echo "FORWARD rules (top 5):"
iptables -L FORWARD -n -v | head -7
echo ""
echo "Now run the robot setup script on the robot:"
echo "  sudo bash robot_network_setup.sh"
echo "=========================================="
