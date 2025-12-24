#!/usr/bin/env python3
"""
Simple test to verify Unitree DDS connection is working

IMPORTANT: Run this in a conda environment with Python 3.8:
    - Locally: conda activate robomimic  (Python 3.8)
    - Docker:  conda activate gmr        (Python 3.10, but unitree_sdk2py may need 3.8)
    
    export LD_LIBRARY_PATH=/home/robo/CodeSpace/twist2_docker/unitree_sdk2/thirdparty/lib/x86_64:$LD_LIBRARY_PATH
    unset CYCLONEDDS_HOME
    python test_unitree_connection.py --net enp4s0
"""
import sys
import time
import os

# Check Python version - unitree_sdk2py requires Python 3.8 (cyclonedds compatibility)
python_version = sys.version_info
print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")

# Check Python version - unitree_sdk2py typically requires Python 3.8
# But in Docker, gmr uses Python 3.10 - we'll allow it but warn
if python_version.major != 3:
    print(f"✗ ERROR: Python {python_version.major} detected (need Python 3)")
    sys.exit(1)

if python_version.minor != 8:
    print(f"⚠ WARNING: Python {python_version.major}.{python_version.minor} detected")
    print(f"  unitree_sdk2py/cyclonedds typically works best with Python 3.8")
    if python_version.minor == 10:
        print(f"  You're using Python 3.10 (likely 'gmr' in Docker) - this may work")
    else:
        print(f"  For local use, try: conda activate robomimic  (has Python 3.8)")
    print()
    print("Attempting to continue... (may fail with import errors)")
    print()

# Add paths for unitree_sdk2py (RoboMimic's version)
script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.abspath(os.path.join(script_dir, '../..'))

# Set up library paths for CycloneDDS BEFORE importing anything
# The libddsc.so is in unitree_sdk2/thirdparty/lib/x86_64/
thirdparty_lib_path = os.path.join(workspace_root, 'unitree_sdk2/thirdparty/lib/x86_64')
if os.path.exists(thirdparty_lib_path):
    # Set LD_LIBRARY_PATH (must be done before importing cyclonedds)
    ld_library_path = os.environ.get('LD_LIBRARY_PATH', '')
    if thirdparty_lib_path not in ld_library_path:
        new_ld_path = f"{thirdparty_lib_path}:{ld_library_path}" if ld_library_path else thirdparty_lib_path
        os.environ['LD_LIBRARY_PATH'] = new_ld_path
        print(f"✓ Set LD_LIBRARY_PATH to include: {thirdparty_lib_path}")
    
    # Unset or override CYCLONEDDS_HOME if it points to wrong location
    # cyclonedds will then use LD_LIBRARY_PATH to find the library
    if os.environ.get('CYCLONEDDS_HOME') == '/usr/local':
        # Unset it so cyclonedds uses LD_LIBRARY_PATH instead
        if 'CYCLONEDDS_HOME' in os.environ:
            del os.environ['CYCLONEDDS_HOME']
        print("✓ Unset CYCLONEDDS_HOME (was pointing to /usr/local)")
    
    # Also try to preload the library using ctypes (before cyclonedds tries to load it)
    try:
        import ctypes
        lib_path = os.path.join(thirdparty_lib_path, 'libddsc.so')
        if os.path.exists(lib_path):
            # Try to load it globally so cyclonedds can find it
            ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
            print(f"✓ Preloaded CycloneDDS library: {lib_path}")
    except Exception as e:
        print(f"⚠ Could not preload libddsc.so: {e}")
        print(f"  Will rely on LD_LIBRARY_PATH")
else:
    print(f"⚠ CycloneDDS library path not found: {thirdparty_lib_path}")

# Try multiple paths
possible_paths = [
    "/workspace/RoboMimic_Deploy/unitree_sdk2_python",  # Docker RoboMimic path
    "/workspace/unitree_sdk2/python_binding",  # Docker unitree_sdk2 path
    os.path.join(workspace_root, "RoboMimic_Deploy/unitree_sdk2_python"),  # Local RoboMimic
    os.path.join(workspace_root, "unitree_sdk2/python_binding"),  # Local unitree_sdk2
    # Also check for unitree_sdk2_python in other locations
    "/home/robo/CodeSpace/hands/inspire_hand_ws/unitree_sdk2_python",  # From error message
]

for path in possible_paths:
    abs_path = os.path.abspath(path)
    if os.path.exists(abs_path):
        sys.path.insert(0, abs_path)
        if "/RoboMimic_Deploy" in abs_path:
            # Also add parent directory for RoboMimic imports
            parent = os.path.dirname(abs_path)
            if parent not in sys.path:
                sys.path.insert(0, parent)

# Try to import Unitree SDK
try:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
    print("✓ Unitree SDK (unitree_sdk2py) imported successfully")
except ImportError as e:
    print(f"✗ Failed to import Unitree SDK: {e}")
    print(f"  Python version: {python_version.major}.{python_version.minor}")
    print(f"  Python path: {sys.path[:5]}")
    print(f"  Tried paths: {possible_paths}")
    print()
    print("Troubleshooting:")
    print("  1. Make sure you're in 'gmr' conda environment:")
    print("     conda activate gmr")
    print("  2. Check if RoboMimic_Deploy is mounted/available")
    print("  3. Try installing unitree_sdk2py:")
    print("     cd /workspace/RoboMimic_Deploy/unitree_sdk2_python && pip install .")
    sys.exit(1)

def test_connection(net="enp4s0", timeout=10):
    """Test DDS connection to robot"""
    print(f"\n{'='*60}")
    print(f"Testing Unitree DDS Connection")
    print(f"{'='*60}")
    print(f"Network interface: {net}")
    print(f"Timeout: {timeout} seconds")
    print()
    
    message_count = 0
    first_message_time = None
    
    def handler(msg):
        nonlocal message_count, first_message_time
        message_count += 1
        if first_message_time is None:
            first_message_time = time.time()
            print(f"✓ FIRST MESSAGE RECEIVED! (tick={msg.tick if hasattr(msg, 'tick') else 'N/A'})")
            print(f"  wireless_remote length: {len(msg.wireless_remote)}")
    
    try:
        print("1. Initializing ChannelFactory...")
        ChannelFactoryInitialize(0, net)
        print("   ✓ ChannelFactory initialized")
        
        print("2. Creating subscriber...")
        subscriber = ChannelSubscriber("rt/lowstate", LowStateHG)
        print("   ✓ Subscriber created")
        
        print("3. Initializing callback...")
        subscriber.Init(handler, 10)
        print("   ✓ Callback initialized")
        print()
        print("4. Waiting for messages from robot...")
        print("   Make sure:")
        print("   - Robot is powered on")
        print("   - Robot is in debugging mode (hold L2+R2 on remote)")
        print("   - Ethernet cable is connected")
        print()
        
        start_time = time.time()
        check_interval = 1.0
        last_check = start_time
        
        while (time.time() - start_time) < timeout:
            time.sleep(0.1)
            
            # Print status every second
            if time.time() - last_check >= check_interval:
                elapsed = time.time() - start_time
                print(f"   [{elapsed:.1f}s] Messages received: {message_count}", end="\r")
                last_check = time.time()
        
        print()  # New line after status updates
        print()
        
        if message_count > 0:
            print(f"✓ SUCCESS! Received {message_count} message(s)")
            if first_message_time:
                latency = first_message_time - start_time
                print(f"  First message latency: {latency:.2f} seconds")
            return True
        else:
            print(f"✗ FAILED: No messages received after {timeout} seconds")
            print()
            print("Troubleshooting:")
            print("  1. Check robot is powered on: Look for LED indicators")
            print("  2. Enter debugging mode: Hold L2+R2 on remote controller")
            print("  3. Check network connection:")
            print(f"     ping 192.168.123.164")
            print("  4. Verify network interface:")
            print(f"     ifconfig {net}")
            print("  5. Check if robot is sending DDS messages:")
            print("     (Robot should publish to 'rt/lowstate' topic)")
            return False
            
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Test Unitree DDS connection",
        epilog="IMPORTANT: Run this in the 'gmr' conda environment:\n"
               "  conda activate gmr\n"
               "  python test_unitree_connection.py --net enp4s0",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--net", type=str, default="enp4s0", 
                       help="Network interface (default: enp4s0)")
    parser.add_argument("--timeout", type=int, default=10,
                       help="Timeout in seconds (default: 10)")
    args = parser.parse_args()
    
    print("="*60)
    print("Unitree DDS Connection Test")
    print("="*60)
    print(f"Python: {sys.version.split()[0]}")
    print(f"Network interface: {args.net}")
    print("="*60)
    print()
    
    success = test_connection(args.net, args.timeout)
    sys.exit(0 if success else 1)

