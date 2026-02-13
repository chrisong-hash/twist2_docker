#!/bin/bash
# Deploy robot peripherals scripts to Unitree robot
# Usage: ./deploy_to_robot.sh [robot_ip]
#
# This script handles complete setup for a new robot:
#   1. SSH key setup (generates if needed, copies to robot)
#   2. Copy peripheral scripts to robot
#   3. Set executable permissions
#   3.5. Create neck_config.json template (if not exists)
#   3.6. WiFi setup (optional, if WIFI_SSID and WIFI_PASSWORD are set)
#   4. Install Python dependencies (dynamixel-sdk, redis, numpy)
#   5. Serial port permissions (adds user to dialout group for /dev/ttyUSB*)
#
# First run on new robot will ask for password ONCE for SSH key copy.
# All subsequent runs are passwordless.
#
# Default Unitree password: 123
#
# WiFi Setup (optional):
#   WIFI_SSID="YourNetwork" WIFI_PASSWORD="YourPassword" ./deploy_to_robot.sh 192.168.123.164

ROBOT_USER="unitree"
ROBOT_IP="${1:-192.168.123.164}"
ROBOT_DEST="/home/unitree"
NEEDS_RELOGIN=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================"
echo "  Deploying to ${ROBOT_USER}@${ROBOT_IP}"
echo "========================================"

# --- SSH Key Setup ---
echo ""
echo "[SSH] Step 1: Checking local SSH key..."
if [ -f ~/.ssh/id_rsa.pub ]; then
    echo "  ✓ SSH key exists (~/.ssh/id_rsa.pub) - skipping generation"
else
    echo "  ✗ No SSH key found, generating..."
    mkdir -p ~/.ssh
    chmod 700 ~/.ssh
    ssh-keygen -t rsa -N "" -f ~/.ssh/id_rsa -q
    chmod 600 ~/.ssh/id_rsa
    chmod 644 ~/.ssh/id_rsa.pub
    echo "  ✓ SSH key generated"
fi

echo ""
echo "[SSH] Step 2: Checking if key is on robot..."
ssh -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=no ${ROBOT_USER}@${ROBOT_IP} exit 2>/dev/null
if [ $? -eq 0 ]; then
    echo "  ✓ SSH key already on robot - skipping copy"
else
    echo "  ✗ SSH key not on robot, need to copy"
    echo ""
    echo "  [!] Default Unitree password is: 123"
    echo ""
    ssh-copy-id -o StrictHostKeyChecking=no ${ROBOT_USER}@${ROBOT_IP}
    if [ $? -ne 0 ]; then
        echo ""
        echo "  ✗ ERROR: Failed to copy SSH key"
        echo "    - Check the password (default: 123)"
        echo "    - Check robot IP: ${ROBOT_IP}"
        echo "    - Check network connection"
        exit 1
    fi
    echo "  ✓ SSH key copied successfully"
fi

echo ""

# Files to deploy
FILES=(
    "robot_peripherals.py"
    "read_neck_position.py"
    "run_peripherals.sh"
    "reset_zed_usb.sh"
)

# Copy files
echo ""
echo "[DEPLOY] Step 3: Copying files to robot..."
for file in "${FILES[@]}"; do
    if [ -f "${SCRIPT_DIR}/${file}" ]; then
        echo "  → ${file}"
        scp -q "${SCRIPT_DIR}/${file}" "${ROBOT_USER}@${ROBOT_IP}:${ROBOT_DEST}/"
    else
        echo "  ✗ ${file} (not found in ${SCRIPT_DIR})"
    fi
done

# Make scripts executable
echo "  → Setting executable permissions..."
ssh ${ROBOT_USER}@${ROBOT_IP} "chmod +x ${ROBOT_DEST}/*.sh ${ROBOT_DEST}/*.py 2>/dev/null"
echo "  ✓ Files deployed"

# Create neck config template if it doesn't exist
echo ""
echo "[CONFIG] Step 3.5: Checking neck calibration config..."
CONFIG_EXISTS=$(ssh ${ROBOT_USER}@${ROBOT_IP} "[ -f ${ROBOT_DEST}/neck_config.json ] && echo yes || echo no")
if [ "$CONFIG_EXISTS" = "yes" ]; then
    echo "  ✓ neck_config.json exists - keeping current values"
    # Show current values
    ssh ${ROBOT_USER}@${ROBOT_IP} "cat ${ROBOT_DEST}/neck_config.json" 2>/dev/null | head -6
else
    echo "  → Creating neck_config.json template..."
    ssh ${ROBOT_USER}@${ROBOT_IP} "cat > ${ROBOT_DEST}/neck_config.json << 'CONFIGEOF'
{
    \"_comment\": \"Run 'python3 read_neck_position.py' to find your calibration values\",
    \"yaw_center\": 2048,
    \"pitch_center\": 2048,
    \"redis_host\": \"192.168.50.164\",
    \"dynamixel_dev\": \"/dev/ttyUSB0\"
}
CONFIGEOF"
    echo "  ✓ Template created"
    echo ""
    echo "  ⚠️  CALIBRATION NEEDED!"
    echo "     1. On robot: python3 read_neck_position.py"
    echo "     2. Note the YAW and PITCH values with head centered"
    echo "     3. Update: YAW_CENTER=XXXX PITCH_CENTER=XXXX ./run_peripherals.sh --save-config"
fi

# --- WiFi Setup (Optional) ---
# Configure these or pass via environment variables
WIFI_SSID="${WIFI_SSID:-}"
WIFI_PASSWORD="${WIFI_PASSWORD:-}"

if [ -n "$WIFI_SSID" ] && [ -n "$WIFI_PASSWORD" ]; then
    echo ""
    echo "[WIFI] Step 3.5: Setting up WiFi on robot..."
    
    # Check if already connected to this network
    CURRENT_WIFI=$(ssh ${ROBOT_USER}@${ROBOT_IP} "nmcli -t -f active,ssid dev wifi | grep '^yes' | cut -d: -f2" 2>/dev/null)
    if [ "$CURRENT_WIFI" = "$WIFI_SSID" ]; then
        echo "  ✓ Already connected to ${WIFI_SSID} - skipping"
    else
        echo "  → Configuring WiFi: ${WIFI_SSID}"
        
        # First, check if sudo needs password
        ssh -T ${ROBOT_USER}@${ROBOT_IP} "sudo -n true 2>/dev/null"
        if [ $? -ne 0 ]; then
            echo "  [!] sudo requires password on this robot"
            echo "  [!] You'll be prompted for the robot password (default: 123)"
            echo ""
            # Use -t to allocate terminal for password prompt
            ssh -t ${ROBOT_USER}@${ROBOT_IP} "
                touch ~/.hushlogin 2>/dev/null
                
                # Unblock WiFi
                sudo rfkill unblock wifi
                sudo rfkill unblock all
                
                # Bring up interface
                sudo ip link set wlan0 up
                
                # Enable NetworkManager
                sudo nmcli radio wifi on
                sudo nmcli device set wlan0 managed yes
                
                # Check if connection profile exists
                if nmcli connection show '${WIFI_SSID}' >/dev/null 2>&1; then
                    echo '    → Connection profile exists, updating...'
                    sudo nmcli connection modify '${WIFI_SSID}' wifi-sec.psk '${WIFI_PASSWORD}'
                else
                    echo '    → Creating new connection profile...'
                    sudo nmcli connection add type wifi ifname wlan0 con-name '${WIFI_SSID}' ssid '${WIFI_SSID}'
                    sudo nmcli connection modify '${WIFI_SSID}' wifi-sec.key-mgmt wpa-psk
                    sudo nmcli connection modify '${WIFI_SSID}' wifi-sec.psk '${WIFI_PASSWORD}'
                fi
                
                # Enable autoconnect and connect
                sudo nmcli connection modify '${WIFI_SSID}' connection.autoconnect yes
                sudo nmcli connection up '${WIFI_SSID}'
            "
        else
            # Passwordless sudo - use -T for cleaner output
            ssh -T ${ROBOT_USER}@${ROBOT_IP} "
                # Unblock WiFi
                sudo rfkill unblock wifi 2>/dev/null
                sudo rfkill unblock all 2>/dev/null
                
                # Bring up interface
                sudo ip link set wlan0 up 2>/dev/null
                
                # Enable NetworkManager
                sudo nmcli radio wifi on 2>/dev/null
                sudo nmcli device set wlan0 managed yes 2>/dev/null
                
                # Check if connection profile exists
                if nmcli connection show '${WIFI_SSID}' >/dev/null 2>&1; then
                    echo '    → Connection profile exists, updating...'
                    sudo nmcli connection modify '${WIFI_SSID}' wifi-sec.psk '${WIFI_PASSWORD}'
                else
                    echo '    → Creating new connection profile...'
                    sudo nmcli connection add type wifi ifname wlan0 con-name '${WIFI_SSID}' ssid '${WIFI_SSID}' >/dev/null
                    sudo nmcli connection modify '${WIFI_SSID}' wifi-sec.key-mgmt wpa-psk
                    sudo nmcli connection modify '${WIFI_SSID}' wifi-sec.psk '${WIFI_PASSWORD}'
                fi
                
                # Enable autoconnect and connect
                sudo nmcli connection modify '${WIFI_SSID}' connection.autoconnect yes
                sudo nmcli connection up '${WIFI_SSID}' 2>/dev/null
            "
        fi
        
        # Verify connection
        sleep 2
        NEW_WIFI=$(ssh ${ROBOT_USER}@${ROBOT_IP} "nmcli -t -f active,ssid dev wifi | grep '^yes' | cut -d: -f2" 2>/dev/null)
        if [ "$NEW_WIFI" = "$WIFI_SSID" ]; then
            echo "  ✓ Connected to ${WIFI_SSID}"
        else
            echo "  ⚠ WiFi configured but not yet connected (may need reboot)"
        fi
    fi
else
    echo ""
    echo "[WIFI] Skipping WiFi setup (set WIFI_SSID and WIFI_PASSWORD to enable)"
fi
# --- Install Dependencies ---
echo ""
echo "[DEPS] Step 4: Checking Python dependencies on robot..."

# Required packages for peripherals
declare -A PACKAGES
PACKAGES=(
    ["dynamixel-sdk"]="Neck servo control"
    ["redis"]="Redis communication"
    ["numpy"]="Array operations"
)

for pkg in "${!PACKAGES[@]}"; do
    desc="${PACKAGES[$pkg]}"
    ssh ${ROBOT_USER}@${ROBOT_IP} "pip3 show ${pkg} >/dev/null 2>&1"
    if [ $? -eq 0 ]; then
        echo "  ✓ ${pkg} - already installed (${desc})"
    else
        echo "  ✗ ${pkg} - not found, installing (${desc})..."
        ssh ${ROBOT_USER}@${ROBOT_IP} "pip3 install --user ${pkg} -q"
        if [ $? -eq 0 ]; then
            echo "    ✓ Installed successfully"
        else
            echo "    ✗ Failed to install ${pkg}"
        fi
    fi
done

# --- Serial Port Permissions ---
echo ""
echo "[PERMS] Step 5: Checking serial port permissions..."

# Check if user is in dialout group (needed for /dev/ttyUSB*)
IN_DIALOUT=$(ssh ${ROBOT_USER}@${ROBOT_IP} "groups | grep -q dialout && echo yes || echo no")
if [ "$IN_DIALOUT" = "yes" ]; then
    echo "  ✓ User ${ROBOT_USER} is in dialout group - serial access OK"
else
    echo "  ✗ User ${ROBOT_USER} not in dialout group, adding..."
    echo "  → This requires sudo (password: 123)"
    # Use -t to allocate TTY for sudo password prompt
    ssh -t ${ROBOT_USER}@${ROBOT_IP} "sudo usermod -aG dialout ${ROBOT_USER}"
    if [ $? -eq 0 ]; then
        echo "  ✓ Added to dialout group"
        echo "  ⚠ NOTE: You need to logout/login or reboot for this to take effect!"
        NEEDS_RELOGIN=true
    else
        echo "  ✗ Failed to add to dialout group"
        echo "  → Manual fix: ssh to robot and run: sudo usermod -aG dialout unitree"
    fi
fi

echo ""
echo "========================================"
echo "  ✓ Deployment complete!"
echo "========================================"
echo ""
echo "Files deployed to: ${ROBOT_USER}@${ROBOT_IP}:${ROBOT_DEST}/"

if [ "$NEEDS_RELOGIN" = "true" ]; then
    echo ""
    echo "⚠️  IMPORTANT: Serial port permissions were changed!"
    echo "   You must logout/login or reboot the robot before accessing /dev/ttyUSB*"
    echo ""
    echo "   Quick fix (one-time): ssh ${ROBOT_USER}@${ROBOT_IP} 'sudo chmod 666 /dev/ttyUSB*'"
    echo "   Or reboot:            ssh ${ROBOT_USER}@${ROBOT_IP} 'sudo reboot'"
fi

echo ""
echo "On the robot, run:"
echo "  cd ~ && ./run_peripherals.sh"
echo ""
echo "Or manually:"
echo "  python3 robot_peripherals.py --redis <PC_IP>"
echo ""
echo "To test neck:"
echo "  python3 read_neck_position.py"

