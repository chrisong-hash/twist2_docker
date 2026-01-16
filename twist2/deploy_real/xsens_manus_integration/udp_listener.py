import socket
import struct

UDP_IP = "0.0.0.0"  # Listen on all interfaces
UDP_PORT = 9763
BUFFER_SIZE = 4096  # Adjust based on your data size

# XSens segment names (MVN body model) - IDs start at 1, not 0
SEGMENT_NAMES = {
    # Body segments (1-23)
    1: "Pelvis",
    2: "L5",
    3: "L3",
    4: "T12",
    5: "T8",
    6: "Neck",
    7: "Head",
    8: "RightShoulder",
    9: "RightUpperArm",
    10: "RightForeArm",
    11: "RightHand",
    12: "LeftShoulder",
    13: "LeftUpperArm",
    14: "LeftForeArm",
    15: "LeftHand",
    16: "RightUpperLeg",
    17: "RightLowerLeg",
    18: "RightFoot",
    19: "RightToe",
    20: "LeftUpperLeg",
    21: "LeftLowerLeg",
    22: "LeftFoot",
    23: "LeftToe",
    
    # LEFT hand finger segments (IDs 24-43, 20 segments)
    # Anatomical naming: First=Thumb, Second=Index, Third=Middle, Fourth=Ring, Fifth=Pinky
    24: "LeftCarpus",                      # Wrist
    25: "LeftFirstMetacarpal",             # Thumb metacarpal
    26: "LeftFirstProximalPhalange",       # Thumb proximal
    27: "LeftFirstDistalPhalange",         # Thumb distal (tip)
    28: "LeftSecondMetacarpal",            # Index metacarpal
    29: "LeftSecondProximalPhalange",      # Index proximal
    30: "LeftSecondMiddlePhalange",        # Index middle
    31: "LeftSecondDistalPhalange",        # Index distal (tip)
    32: "LeftThirdMetacarpal",             # Middle metacarpal
    33: "LeftThirdProximalPhalange",       # Middle proximal
    34: "LeftThirdMiddlePhalange",         # Middle middle
    35: "LeftThirdDistalPhalange",         # Middle distal (tip)
    36: "LeftFourthMetacarpal",            # Ring metacarpal
    37: "LeftFourthProximalPhalange",      # Ring proximal
    38: "LeftFourthMiddlePhalange",        # Ring middle
    39: "LeftFourthDistalPhalange",        # Ring distal (tip)
    40: "LeftFifthMetacarpal",             # Pinky metacarpal
    41: "LeftFifthProximalPhalange",       # Pinky proximal
    42: "LeftFifthMiddlePhalange",         # Pinky middle
    43: "LeftFifthDistalPhalange",         # Pinky distal (tip)
    
    # RIGHT hand finger segments (IDs 44-63, 20 segments)
    44: "RightCarpus",                     # Wrist
    45: "RightFirstMetacarpal",            # Thumb metacarpal
    46: "RightFirstProximalPhalange",      # Thumb proximal
    47: "RightFirstDistalPhalange",        # Thumb distal (tip)
    48: "RightSecondMetacarpal",           # Index metacarpal
    49: "RightSecondProximalPhalange",     # Index proximal
    50: "RightSecondMiddlePhalange",       # Index middle
    51: "RightSecondDistalPhalange",       # Index distal (tip)
    52: "RightThirdMetacarpal",            # Middle metacarpal
    53: "RightThirdProximalPhalange",      # Middle proximal
    54: "RightThirdMiddlePhalange",        # Middle middle
    55: "RightThirdDistalPhalange",        # Middle distal (tip)
    56: "RightFourthMetacarpal",           # Ring metacarpal
    57: "RightFourthProximalPhalange",     # Ring proximal
    58: "RightFourthMiddlePhalange",       # Ring middle
    59: "RightFourthDistalPhalange",       # Ring distal (tip)
    60: "RightFifthMetacarpal",            # Pinky metacarpal
    61: "RightFifthProximalPhalange",      # Pinky proximal
    62: "RightFifthMiddlePhalange",        # Pinky middle
    63: "RightFifthDistalPhalange",        # Pinky distal (tip)
}

# Helper sets to identify body vs finger segments
# Left hand: IDs 24-43 (20 segments), Right hand: IDs 44-63 (20 segments)
BODY_SEGMENT_IDS = set(range(1, 24))  # IDs 1-23 (body)
LEFT_FINGER_SEGMENT_IDS = set(range(24, 44))   # IDs 24-43 (LEFT hand - 20 segments)
RIGHT_FINGER_SEGMENT_IDS = set(range(44, 64))  # IDs 44-63 (RIGHT hand - 20 segments)
FINGER_SEGMENT_IDS = LEFT_FINGER_SEGMENT_IDS | RIGHT_FINGER_SEGMENT_IDS


def decode_xsens_packet(data):
    """Decode XSens MVN UDP packet (MXTP02 format - Quaternion + Position)
    
    Returns dict with:
        - 'segments': All segments (body + fingers)
        - 'body_segments': Only body segments (IDs 1-24)
        - 'finger_segments': Only finger segments (IDs 25-64)
        - 'left_finger_segments': Left hand finger segments (IDs 45-64)
        - 'right_finger_segments': Right hand finger segments (IDs 25-44)
    """
    
    if len(data) < 24:
        return None
    
    # Check header
    header_id = data[0:6].decode('ascii', errors='ignore')
    if not header_id.startswith("MXTP"):
        print(f"Unknown packet type: {header_id}")
        return None
    
    message_type = header_id[4:6]
    
    # Parse header (24 bytes total)
    # Bytes 6-9: Sample counter (big-endian uint32)
    sample_counter = struct.unpack('>I', data[6:10])[0]
    
    # Byte 10: Datagram counter
    datagram_counter = data[10]
    
    # Byte 11: Number of segments
    num_segments = data[11]
    
    # Bytes 12-15: Time code
    time_code = struct.unpack('>I', data[12:16])[0]
    
    # Byte 16: Character ID
    character_id = data[16]
    
    # Bytes 17-23: Reserved
    
    result = {
        'header': header_id,
        'message_type': message_type,
        'sample_counter': sample_counter,
        'datagram_counter': datagram_counter,
        'num_segments': num_segments,
        'time_code': time_code,
        'character_id': character_id,
        'segments': [],
        'body_segments': [],
        'finger_segments': [],
        'left_finger_segments': [],
        'right_finger_segments': [],
    }
    
    # Parse segment data starting at byte 24
    offset = 24
    
    if message_type == "02":
        # Type 02: Segment ID (4 bytes) + Position (3 floats) + Quaternion (4 floats)
        # = 4 + 12 + 16 = 32 bytes per segment
        # NOTE: Position comes BEFORE Quaternion (verified by checking quaternion norm)
        segment_size = 32
        
        for i in range(num_segments):
            if offset + segment_size > len(data):
                break
            
            segment_data = data[offset:offset + segment_size]
            
            # Segment ID (4 bytes, big-endian)
            segment_id = struct.unpack('>I', segment_data[0:4])[0]
            
            # Position (3 floats, big-endian): x, y, z - bytes 4-16
            pos_x = struct.unpack('>f', segment_data[4:8])[0]
            pos_y = struct.unpack('>f', segment_data[8:12])[0]
            pos_z = struct.unpack('>f', segment_data[12:16])[0]
            
            # Quaternion (4 floats, big-endian): w, x, y, z - bytes 16-32
            q0 = struct.unpack('>f', segment_data[16:20])[0]  # w
            q1 = struct.unpack('>f', segment_data[20:24])[0]  # x
            q2 = struct.unpack('>f', segment_data[24:28])[0]  # y
            q3 = struct.unpack('>f', segment_data[28:32])[0]  # z
            
            segment_name = SEGMENT_NAMES.get(segment_id, f"Segment_{segment_id}")
            
            segment_entry = {
                'id': segment_id,
                'name': segment_name,
                'quaternion': (q0, q1, q2, q3),  # w, x, y, z format
                'position': (pos_x, pos_y, pos_z)
            }
            
            # Add to all segments list
            result['segments'].append(segment_entry)
            
            # Categorize by type
            if segment_id in BODY_SEGMENT_IDS:
                result['body_segments'].append(segment_entry)
            elif segment_id in RIGHT_FINGER_SEGMENT_IDS:
                result['finger_segments'].append(segment_entry)
                result['right_finger_segments'].append(segment_entry)
            elif segment_id in LEFT_FINGER_SEGMENT_IDS:
                result['finger_segments'].append(segment_entry)
                result['left_finger_segments'].append(segment_entry)
            
            offset += segment_size
    
    return result


def print_decoded_data(decoded, show_fingers=True):
    """Pretty print decoded XSens data
    
    Args:
        decoded: Decoded packet data
        show_fingers: If True, show finger segment data (default True)
    """
    if decoded is None:
        return
    
    num_body = len(decoded['body_segments'])
    num_fingers = len(decoded['finger_segments'])
    num_left = len(decoded['left_finger_segments'])
    num_right = len(decoded['right_finger_segments'])
    
    print(f"\n{'='*80}")
    print(f"Header: {decoded['header']} | Sample: {decoded['sample_counter']} | "
          f"Time: {decoded['time_code']} | Total Segments: {decoded['num_segments']}")
    print(f"Body: {num_body} | Fingers: {num_fingers} (L:{num_left} R:{num_right})")
    print(f"{'='*80}")
    
    # Print body segments
    if decoded['body_segments']:
        print("\n[BODY SEGMENTS]")
        for seg in decoded['body_segments']:
            q = seg['quaternion']
            p = seg['position']
            print(f"  {seg['name']:20} | "
                  f"Quat(w={q[0]:7.4f}, x={q[1]:7.4f}, y={q[2]:7.4f}, z={q[3]:7.4f}) | "
                  f"Pos({p[0]:7.3f}, {p[1]:7.3f}, {p[2]:7.3f})")
    
    # Print finger segments
    if show_fingers and decoded['finger_segments']:
        if decoded['right_finger_segments']:
            print("\n[RIGHT HAND FINGERS]")
            for seg in decoded['right_finger_segments']:
                q = seg['quaternion']
                p = seg['position']
                print(f"  {seg['name']:20} | "
                      f"Quat(w={q[0]:7.4f}, x={q[1]:7.4f}, y={q[2]:7.4f}, z={q[3]:7.4f}) | "
                      f"Pos({p[0]:7.3f}, {p[1]:7.3f}, {p[2]:7.3f})")
        
        if decoded['left_finger_segments']:
            print("\n[LEFT HAND FINGERS]")
            for seg in decoded['left_finger_segments']:
                q = seg['quaternion']
                p = seg['position']
                print(f"  {seg['name']:20} | "
                      f"Quat(w={q[0]:7.4f}, x={q[1]:7.4f}, y={q[2]:7.4f}, z={q[3]:7.4f}) | "
                      f"Pos({p[0]:7.3f}, {p[1]:7.3f}, {p[2]:7.3f})")


def get_finger_data_by_name(decoded):
    """Extract finger segments organized by finger name for easier access.
    
    Uses Manus Pro anatomical naming: CMC, MCP, PIP, DIP, IP (thumb)
    
    Returns dict like:
        {
            'left': {
                'wrist': segment or None,
                'thumb': [CMC, MCP, IP segments],
                'index': [MC, MCP, PIP, DIP segments],
                'middle': [MC, MCP, PIP, DIP segments],
                'ring': [MC, MCP, PIP, DIP segments],
                'pinky': [MC, MCP, PIP segments],  # Left pinky missing DIP
            },
            'right': {
                'wrist': segment or None,
                'thumb': [CMC, MCP, IP segments],
                'index': [MC, MCP, PIP, DIP segments],
                'middle': [MC, MCP, PIP, DIP segments],
                'ring': [MC, MCP, PIP, DIP segments],
                'pinky': [MC, MCP, PIP, DIP segments],
            }
        }
    """
    result = {
        'left': {'wrist': None, 'thumb': [], 'index': [], 'middle': [], 'ring': [], 'pinky': []},
        'right': {'wrist': None, 'thumb': [], 'index': [], 'middle': [], 'ring': [], 'pinky': []},
    }
    
    finger_map = {
        # Left hand (IDs 24-43) - First=Thumb, Second=Index, Third=Middle, Fourth=Ring, Fifth=Pinky
        'LeftCarpus': ('left', 'wrist'),
        'LeftFirstMetacarpal': ('left', 'thumb'), 'LeftFirstProximalPhalange': ('left', 'thumb'), 'LeftFirstDistalPhalange': ('left', 'thumb'),
        'LeftSecondMetacarpal': ('left', 'index'), 'LeftSecondProximalPhalange': ('left', 'index'), 'LeftSecondMiddlePhalange': ('left', 'index'), 'LeftSecondDistalPhalange': ('left', 'index'),
        'LeftThirdMetacarpal': ('left', 'middle'), 'LeftThirdProximalPhalange': ('left', 'middle'), 'LeftThirdMiddlePhalange': ('left', 'middle'), 'LeftThirdDistalPhalange': ('left', 'middle'),
        'LeftFourthMetacarpal': ('left', 'ring'), 'LeftFourthProximalPhalange': ('left', 'ring'), 'LeftFourthMiddlePhalange': ('left', 'ring'), 'LeftFourthDistalPhalange': ('left', 'ring'),
        'LeftFifthMetacarpal': ('left', 'pinky'), 'LeftFifthProximalPhalange': ('left', 'pinky'), 'LeftFifthMiddlePhalange': ('left', 'pinky'), 'LeftFifthDistalPhalange': ('left', 'pinky'),
        # Right hand (IDs 44-63)
        'RightCarpus': ('right', 'wrist'),
        'RightFirstMetacarpal': ('right', 'thumb'), 'RightFirstProximalPhalange': ('right', 'thumb'), 'RightFirstDistalPhalange': ('right', 'thumb'),
        'RightSecondMetacarpal': ('right', 'index'), 'RightSecondProximalPhalange': ('right', 'index'), 'RightSecondMiddlePhalange': ('right', 'index'), 'RightSecondDistalPhalange': ('right', 'index'),
        'RightThirdMetacarpal': ('right', 'middle'), 'RightThirdProximalPhalange': ('right', 'middle'), 'RightThirdMiddlePhalange': ('right', 'middle'), 'RightThirdDistalPhalange': ('right', 'middle'),
        'RightFourthMetacarpal': ('right', 'ring'), 'RightFourthProximalPhalange': ('right', 'ring'), 'RightFourthMiddlePhalange': ('right', 'ring'), 'RightFourthDistalPhalange': ('right', 'ring'),
        'RightFifthMetacarpal': ('right', 'pinky'), 'RightFifthProximalPhalange': ('right', 'pinky'), 'RightFifthMiddlePhalange': ('right', 'pinky'), 'RightFifthDistalPhalange': ('right', 'pinky'),
    }
    
    for seg in decoded.get('finger_segments', []):
        name = seg['name']
        if name in finger_map:
            hand, finger = finger_map[name]
            if finger == 'wrist':
                result[hand][finger] = seg
            else:
                result[hand][finger].append(seg)
    
    return result


def print_finger_summary(decoded):
    """Print a compact summary of finger data (useful for debugging)."""
    finger_data = get_finger_data_by_name(decoded)
    
    for hand in ['right', 'left']:
        hand_data = finger_data[hand]
        if not any(hand_data[f] for f in ['thumb', 'index', 'middle', 'ring', 'pinky']):
            continue
        
        print(f"\n[{hand.upper()} HAND SUMMARY]")
        for finger in ['thumb', 'index', 'middle', 'ring', 'pinky']:
            segs = hand_data[finger]
            if segs:
                # Just show the quaternion W component as a simple "curl" indicator
                curls = [f"{s['quaternion'][0]:.2f}" for s in segs]
                print(f"  {finger:8}: {' -> '.join(curls)}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="XSens MVN UDP Listener with Manus Glove Support")
    parser.add_argument("--port", type=int, default=UDP_PORT, help=f"UDP port (default: {UDP_PORT})")
    parser.add_argument("--ip", type=str, default=UDP_IP, help=f"IP to bind (default: {UDP_IP})")
    parser.add_argument("--body-only", action="store_true", help="Only show body segments")
    parser.add_argument("--fingers-only", action="store_true", help="Only show finger segments")
    parser.add_argument("--summary", action="store_true", help="Show compact finger summary")
    parser.add_argument("--quiet", action="store_true", help="Minimal output (just segment counts)")
    args = parser.parse_args()
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.ip, args.port))
    
    print(f"Listening for XSens UDP packets on {args.ip}:{args.port}...")
    print("Supports body segments (1-24) + Manus glove finger segments (25-64)")
    print("Press Ctrl+C to stop\n")
    
    try:
        while True:
            data, addr = sock.recvfrom(BUFFER_SIZE)
            
            decoded = decode_xsens_packet(data)
            if decoded:
                if args.quiet:
                    # Minimal output
                    num_body = len(decoded['body_segments'])
                    num_fingers = len(decoded['finger_segments'])
                    print(f"Sample {decoded['sample_counter']}: Body={num_body}, Fingers={num_fingers}")
                elif args.summary:
                    # Compact finger summary
                    print(f"\nSample {decoded['sample_counter']}:")
                    print_finger_summary(decoded)
                elif args.body_only:
                    # Only body
                    print(f"\nReceived {len(data)} bytes from {addr[0]}:{addr[1]}")
                    print_decoded_data(decoded, show_fingers=False)
                elif args.fingers_only:
                    # Only fingers
                    num_fingers = len(decoded['finger_segments'])
                    if num_fingers > 0:
                        print(f"\n[Sample {decoded['sample_counter']}] Finger segments: {num_fingers}")
                        print_finger_summary(decoded)
                    else:
                        print(f"Sample {decoded['sample_counter']}: No finger data")
                else:
                    # Full output
                    print(f"\nReceived {len(data)} bytes from {addr[0]}:{addr[1]}")
                    print_decoded_data(decoded, show_fingers=True)
            else:
                print(f"Unknown packet ({len(data)} bytes): {data[:20].hex()}...")
                
    except KeyboardInterrupt:
        print("\nStopping listener...")
    finally:
        sock.close()


if __name__ == "__main__":
    main()