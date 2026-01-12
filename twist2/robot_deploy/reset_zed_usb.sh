#!/bin/bash
# Reset ZED camera USB device without physical unplug
# Run with: sudo ./reset_zed_usb.sh
# 
# Tries multiple methods in order of aggressiveness

set -e

echo "=========================================="
echo "  ZED Camera USB Reset"
echo "=========================================="

# Find ZED USB device
echo "[1] Looking for ZED camera..."
ZED_LINE=$(lsusb | grep -iE "ZED|2b03:" || true)

if [ -z "$ZED_LINE" ]; then
    echo "    ZED camera not found in lsusb"
    echo "    It may already be disconnected/failed"
else
    echo "    Found: $ZED_LINE"
fi

# Extract bus and device number
BUS=$(echo "$ZED_LINE" | awk '{print $2}' || echo "")
DEV=$(echo "$ZED_LINE" | awk '{print $4}' | tr -d ':' || echo "")

# Method 1: uhubctl power cycle (most reliable if available)
echo ""
echo "[2] Trying uhubctl power cycle..."
if command -v uhubctl &> /dev/null; then
    # Find which hub/port the ZED is on and power cycle it
    uhubctl -a cycle -l 1 2>/dev/null || uhubctl -a cycle 2>/dev/null || echo "    uhubctl failed"
    sleep 3
else
    echo "    uhubctl not installed (install with: sudo apt install uhubctl)"
fi

# Method 2: USB authorize toggle
echo ""
echo "[3] Trying USB authorize toggle..."
for auth_file in /sys/bus/usb/devices/*/authorized; do
    dev_dir=$(dirname "$auth_file")
    if [ -f "$dev_dir/product" ]; then
        product=$(cat "$dev_dir/product" 2>/dev/null || echo "")
        if echo "$product" | grep -qi "ZED"; then
            echo "    Found ZED at $dev_dir"
            echo "    Deauthorizing..."
            echo 0 | sudo tee "$auth_file" > /dev/null
            sleep 1
            echo "    Reauthorizing..."
            echo 1 | sudo tee "$auth_file" > /dev/null
            sleep 2
            echo "    Done"
        fi
    fi
done

# Method 3: Unbind/rebind USB driver
echo ""
echo "[4] Trying USB driver unbind/rebind..."
for dev in /sys/bus/usb/devices/*; do
    if [ -f "$dev/product" ]; then
        product=$(cat "$dev/product" 2>/dev/null || echo "")
        if echo "$product" | grep -qi "ZED"; then
            dev_id=$(basename "$dev")
            echo "    Found ZED device: $dev_id"
            
            # Unbind from driver
            if [ -f "/sys/bus/usb/drivers/usb/$dev_id" ] || [ -L "/sys/bus/usb/drivers/usb/$dev_id" ]; then
                echo "    Unbinding $dev_id..."
                echo "$dev_id" | sudo tee /sys/bus/usb/drivers/usb/unbind > /dev/null 2>&1 || true
                sleep 2
                echo "    Rebinding $dev_id..."
                echo "$dev_id" | sudo tee /sys/bus/usb/drivers/usb/bind > /dev/null 2>&1 || true
                sleep 2
            fi
        fi
    fi
done

# Method 4: Reset USB controller (nuclear option)
echo ""
echo "[5] Trying USB controller reset..."
# Find the USB controller for the ZED
if [ -n "$BUS" ]; then
    USB_CONTROLLER=$(readlink -f /sys/bus/usb/devices/usb$BUS 2>/dev/null | sed 's|.*devices/||; s|/usb.*||' || echo "")
    if [ -n "$USB_CONTROLLER" ] && [ -f "/sys/bus/pci/devices/$USB_CONTROLLER/reset" ]; then
        echo "    Resetting USB controller: $USB_CONTROLLER"
        echo 1 | sudo tee "/sys/bus/pci/devices/$USB_CONTROLLER/reset" > /dev/null 2>&1 || echo "    Reset failed"
        sleep 3
    else
        echo "    Could not find USB controller reset file"
    fi
fi

# Method 5: Reload USB kernel modules (most aggressive)
echo ""
echo "[6] Would you like to reload USB kernel modules? (RISKY - may affect other USB devices)"
echo "    To do this manually, run:"
echo "      sudo modprobe -r xhci_pci && sudo modprobe xhci_pci"
echo ""

# Check if ZED is back
echo "=========================================="
echo "  Checking result..."
echo "=========================================="
sleep 2
NEW_ZED=$(lsusb | grep -iE "ZED|2b03:" || true)
if [ -n "$NEW_ZED" ]; then
    echo "[✓] ZED camera detected: $NEW_ZED"
else
    echo "[✗] ZED camera still not detected"
    echo ""
    echo "If nothing worked, try:"
    echo "  1. sudo modprobe -r xhci_pci && sleep 2 && sudo modprobe xhci_pci"
    echo "  2. Physical unplug/replug"
    echo "  3. Check USB cable and port"
fi
