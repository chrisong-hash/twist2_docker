#!/usr/bin/env python3
"""
Simple test script for Inspire hands.
Tests basic communication and open/close functionality.
"""

import sys
import time
sys.path.insert(0, '/workspace/twist2/deploy_real')

from robot_control.inspire_hand_wrapper import DualHandController

LEFT_IP = "192.168.123.210"
RIGHT_IP = "192.168.123.211"

def main():
    print("=" * 50)
    print("🤖 Inspire Hand Test")
    print("=" * 50)
    print(f"  Left hand:  {LEFT_IP}")
    print(f"  Right hand: {RIGHT_IP}")
    print("=" * 50)
    print()
    
    # Initialize
    print("[1/4] Connecting to hands...")
    try:
        controller = DualHandController(
            left_ip=LEFT_IP,
            right_ip=RIGHT_IP,
            async_mode=True
        )
        print("  ✓ Hands connected")
    except Exception as e:
        print(f"  ✗ Failed to connect: {e}")
        return 1
    
    # Test open
    print("\n[2/4] Opening both hands...")
    try:
        controller.ctrl_dual_hand(0.0, 0.0)  # 0 = open
        print("  → Sent OPEN command")
        time.sleep(2)
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Test close
    print("\n[3/4] Closing both hands...")
    try:
        controller.ctrl_dual_hand(1.0, 1.0)  # 1 = closed
        print("  → Sent CLOSE command")
        time.sleep(2)
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Test open again
    print("\n[4/4] Opening both hands again...")
    try:
        controller.ctrl_dual_hand(0.0, 0.0)
        print("  → Sent OPEN command")
        time.sleep(2)
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    print()
    print("=" * 50)
    print("Test complete!")
    print("=" * 50)
    print()
    print("Did the hands move? (open → close → open)")
    print("If not, check:")
    print("  1. Hand power is ON")
    print("  2. Network cables connected")
    print("  3. Hand IPs are correct")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())


