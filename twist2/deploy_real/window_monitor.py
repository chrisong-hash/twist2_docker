#!/usr/bin/env python3 -u
"""
Window Monitor - Automatically moves specified windows to the second monitor.

Runs as a systemd service: mujoco-window-monitor.service

Usage:
    python window_monitor.py &          # Run in background
    python window_monitor.py --list     # List current windows
    python window_monitor.py --detect   # Show monitor info

Service commands:
    systemctl --user status mujoco-window-monitor   # Check status
    systemctl --user restart mujoco-window-monitor  # Restart
    journalctl --user -u mujoco-window-monitor -f   # View logs

Add new window patterns to WINDOW_PATTERNS below.
"""

import sys
import time
import argparse
import subprocess
import re
from typing import List, Tuple, Optional

# Force unbuffered output for systemd journal
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# ============================================================================
# CONFIGURATION - Add your window patterns here!
# ============================================================================
# Each pattern is checked against window names (case-insensitive by default)
# You can use:
#   - Simple strings: "MuJoCo" matches any window containing "MuJoCo"
#   - Regex patterns: r"^Isaac.*Gym$" for more complex matching

WINDOW_PATTERNS = [
    # MuJoCo windows
    "MuJoCo",
    "Simulate",
    "mujoco",
    
    # Isaac Gym / IsaacGym windows
    "Isaac",
    "IsaacGym",
    
    # Add more patterns here as needed:
    # "MyCustomApp",
    # "Gazebo",
    # "RViz",
]

# Monitor selection: "second" uses auto-detected second monitor
# Or specify exact position like (3840, 100)
TARGET_MONITOR = "second"

# Window sizing: percentage of target monitor width (0.0-1.0)
# Height is calculated automatically to preserve aspect ratio
TARGET_WIDTH_PERCENT = 0.80

# Y offset from top of monitor (in pixels)
TARGET_Y_OFFSET = 50

# Polling interval in seconds
POLL_INTERVAL = 0.5

# ============================================================================
# Monitor Detection
# ============================================================================

def get_monitors() -> List[dict]:
    """Detect connected monitors using xrandr."""
    try:
        result = subprocess.run(
            ["xrandr", "--query"],
            capture_output=True, text=True, timeout=5
        )
        monitors = []
        for line in result.stdout.split('\n'):
            if ' connected' in line:
                # Parse: "HDMI-0 connected 2160x3840+3840+0 right"
                # or: "DP-0 connected primary 3840x2160+0+777"
                match = re.search(r'(\S+) connected\s*(primary)?\s*(\d+)x(\d+)\+(\d+)\+(\d+)', line)
                if match:
                    monitors.append({
                        'name': match.group(1),
                        'primary': match.group(2) == 'primary',
                        'width': int(match.group(3)),
                        'height': int(match.group(4)),
                        'x': int(match.group(5)),
                        'y': int(match.group(6)),
                    })
        return monitors
    except Exception as e:
        print(f"Error detecting monitors: {e}")
        return []


def get_second_monitor() -> dict:
    """Get the second (non-primary) monitor info."""
    monitors = get_monitors()
    
    if len(monitors) < 2:
        print("Warning: Only one monitor detected, using defaults")
        return {'x': 1920, 'y': 0, 'width': 1920, 'height': 1080}
    
    # Find non-primary monitor, or the one with highest x position
    non_primary = [m for m in monitors if not m['primary']]
    if non_primary:
        return non_primary[0]
    else:
        # All monitors are primary? Use the one with highest x
        monitors_sorted = sorted(monitors, key=lambda m: m['x'], reverse=True)
        return monitors_sorted[0]


def get_second_monitor_position() -> Tuple[int, int]:
    """Get the position of the second (non-primary) monitor."""
    m = get_second_monitor()
    return (m['x'], m['y'])


def get_target_position() -> Tuple[int, int]:
    """Get the target position based on configuration."""
    if TARGET_MONITOR == "second":
        return get_second_monitor_position()
    elif isinstance(TARGET_MONITOR, tuple):
        return TARGET_MONITOR
    else:
        return get_second_monitor_position()


def get_target_monitor() -> dict:
    """Get the target monitor info based on configuration."""
    if TARGET_MONITOR == "second":
        m = get_second_monitor()
    else:
        m = get_second_monitor()
    
    # Apply custom width percentage and Y offset
    m['width'] = int(m['width'] * TARGET_WIDTH_PERCENT)
    m['y'] = m['y'] + TARGET_Y_OFFSET
    return m


# ============================================================================
# Window Management (using Xlib with _NET_MOVERESIZE_WINDOW for GNOME/Unity)
# ============================================================================

def get_display():
    """Get or create X display connection."""
    from Xlib import display
    return display.Display()


def get_windows() -> List[Tuple[int, str, any]]:
    """Get list of (window_id, window_name, window_obj) using Xlib."""
    try:
        from Xlib import X
        d = get_display()
        root = d.screen().root
        
        # Get window list from _NET_CLIENT_LIST
        atom = d.intern_atom('_NET_CLIENT_LIST')
        resp = root.get_full_property(atom, X.AnyPropertyType)
        
        result = []
        if resp:
            for wid in resp.value:
                try:
                    win = d.create_resource_object('window', wid)
                    name = win.get_wm_name()
                    if name:
                        name_str = name if isinstance(name, str) else str(name)
                        result.append((wid, name_str, win))
                except Exception:
                    pass
        return result
    except ImportError:
        print("Error: python-xlib not installed. Run: pip install python-xlib")
        return []
    except Exception as e:
        print(f"Error getting windows: {e}")
        return []


def unmaximize_window(win) -> bool:
    """Remove maximized/fullscreen state from a window."""
    try:
        from Xlib import X
        from Xlib.protocol import event
        
        d = get_display()
        root = d.screen().root
        
        # Atoms for window states
        NET_WM_STATE = d.intern_atom('_NET_WM_STATE')
        NET_WM_STATE_MAXIMIZED_VERT = d.intern_atom('_NET_WM_STATE_MAXIMIZED_VERT')
        NET_WM_STATE_MAXIMIZED_HORZ = d.intern_atom('_NET_WM_STATE_MAXIMIZED_HORZ')
        NET_WM_STATE_FULLSCREEN = d.intern_atom('_NET_WM_STATE_FULLSCREEN')
        
        # Action: 0 = remove, 1 = add, 2 = toggle
        _NET_WM_STATE_REMOVE = 0
        
        # Remove maximized vertical
        ev = event.ClientMessage(
            window=win,
            client_type=NET_WM_STATE,
            data=(32, [_NET_WM_STATE_REMOVE, NET_WM_STATE_MAXIMIZED_VERT, NET_WM_STATE_MAXIMIZED_HORZ, 0, 0])
        )
        root.send_event(ev, event_mask=X.SubstructureRedirectMask | X.SubstructureNotifyMask)
        
        # Remove fullscreen
        ev = event.ClientMessage(
            window=win,
            client_type=NET_WM_STATE,
            data=(32, [_NET_WM_STATE_REMOVE, NET_WM_STATE_FULLSCREEN, 0, 0, 0])
        )
        root.send_event(ev, event_mask=X.SubstructureRedirectMask | X.SubstructureNotifyMask)
        
        d.sync()
        return True
    except Exception as e:
        print(f"Error unmaximizing window: {e}")
        return False


def get_window_geometry(win) -> Tuple[int, int]:
    """Get window width and height."""
    try:
        geom = win.get_geometry()
        return (geom.width, geom.height)
    except:
        return (800, 600)  # Default fallback


def activate_window(win) -> bool:
    """Activate/raise a window to restore it from minimized/iconified state."""
    try:
        from Xlib import X
        from Xlib.protocol import event
        
        d = get_display()
        root = d.screen().root
        
        # Use _NET_ACTIVE_WINDOW to activate/raise the window
        NET_ACTIVE_WINDOW = d.intern_atom('_NET_ACTIVE_WINDOW')
        ev = event.ClientMessage(
            window=win,
            client_type=NET_ACTIVE_WINDOW,
            data=(32, [2, 0, 0, 0, 0])  # Source indication: 2 = pager/other
        )
        root.send_event(ev, event_mask=X.SubstructureRedirectMask | X.SubstructureNotifyMask)
        d.sync()
        return True
    except Exception as e:
        print(f"Error activating window: {e}")
        return False


def move_window(win, x: int, y: int, target_width: int = 0, preserve_ratio: bool = True) -> bool:
    """Move and optionally resize a window using _NET_MOVERESIZE_WINDOW (works with GNOME/Unity).
    
    If target_width > 0 and preserve_ratio is True, calculates height to maintain aspect ratio.
    """
    try:
        from Xlib import X
        from Xlib.protocol import event
        import time
        
        d = get_display()
        root = d.screen().root
        
        # First, activate the window (restore from minimized/iconified)
        activate_window(win)
        time.sleep(0.1)  # Give WM time to process
        
        # Remove maximized/fullscreen state
        unmaximize_window(win)
        time.sleep(0.1)  # Give WM time to process
        
        # Calculate new dimensions if target_width is specified
        width = 0
        height = 0
        if target_width > 0:
            current_w, current_h = get_window_geometry(win)
            if current_w > 0 and current_h > 0 and preserve_ratio:
                # Calculate height to maintain aspect ratio
                aspect_ratio = current_h / current_w
                width = target_width
                height = int(target_width * aspect_ratio)
            else:
                width = target_width
        
        # Use _NET_MOVERESIZE_WINDOW ClientMessage
        NET_MOVERESIZE_WINDOW = d.intern_atom('_NET_MOVERESIZE_WINDOW')
        
        # Flags: bit 8 = x, bit 9 = y, bit 10 = width, bit 11 = height
        flags = (1 << 8) | (1 << 9)  # x and y
        if width > 0:
            flags |= (1 << 10)  # width
        if height > 0:
            flags |= (1 << 11)  # height
        
        ev = event.ClientMessage(
            window=win,
            client_type=NET_MOVERESIZE_WINDOW,
            data=(32, [flags, x, y, width, height])
        )
        
        root.send_event(ev, event_mask=X.SubstructureRedirectMask | X.SubstructureNotifyMask)
        d.sync()
        return True
    except Exception as e:
        print(f"Error moving window: {e}")
        return False


def matches_pattern(window_name: str, patterns: List[str]) -> bool:
    """Check if window name matches any of the patterns."""
    window_lower = window_name.lower()
    for pattern in patterns:
        pattern_lower = pattern.lower()
        # Try as regex first
        try:
            if re.search(pattern, window_name, re.IGNORECASE):
                return True
        except re.error:
            pass
        # Fall back to simple substring match
        if pattern_lower in window_lower:
            return True
    return False


# ============================================================================
# Main Monitor Loop
# ============================================================================

class WindowMonitor:
    def __init__(self, patterns: List[str], target_monitor: dict):
        self.patterns = patterns
        self.target_x = target_monitor['x']
        self.target_y = target_monitor['y']
        self.target_width = target_monitor['width']
        self.target_height = target_monitor['height']
        self.moved_windows = set()  # Track windows we've already moved
        
    def check_and_move_windows(self) -> int:
        """Check for matching windows and move them. Returns count of moved windows."""
        moved_count = 0
        windows = get_windows()
        
        for win_id, win_name, win_obj in windows:
            # Skip if already moved
            if win_id in self.moved_windows:
                continue
                
            # Check if matches pattern
            if matches_pattern(win_name, self.patterns):
                # Fit width to monitor, preserve aspect ratio
                print(f"[Monitor] Found: '{win_name}' -> Moving to ({self.target_x}, {self.target_y}) width={self.target_width} (preserving ratio)")
                if move_window(win_obj, self.target_x, self.target_y, self.target_width, preserve_ratio=True):
                    self.moved_windows.add(win_id)
                    moved_count += 1
                    
        return moved_count
    
    def run(self, poll_interval: float = POLL_INTERVAL):
        """Main monitoring loop."""
        print(f"[Monitor] Started - watching for windows matching: {self.patterns}")
        print(f"[Monitor] Target: ({self.target_x}, {self.target_y}) width={self.target_width} (aspect ratio preserved)")
        print(f"[Monitor] Press Ctrl+C to stop\n")
        
        try:
            while True:
                self.check_and_move_windows()
                time.sleep(poll_interval)
        except KeyboardInterrupt:
            print("\n[Monitor] Stopped")


# ============================================================================
# CLI
# ============================================================================

def list_windows():
    """List all current windows."""
    print("Current windows:")
    print("-" * 60)
    windows = get_windows()
    for win_id, win_name, _ in windows:
        matches = "✓" if matches_pattern(win_name, WINDOW_PATTERNS) else " "
        print(f"  [{matches}] {win_name}")
    print("-" * 60)
    print(f"Total: {len(windows)} windows")
    print(f"Matching patterns: {sum(1 for _, n, _ in windows if matches_pattern(n, WINDOW_PATTERNS))}")


def show_monitors():
    """Show monitor information."""
    print("Detected monitors:")
    print("-" * 60)
    monitors = get_monitors()
    for m in monitors:
        primary = " (primary)" if m['primary'] else ""
        print(f"  {m['name']}{primary}: {m['width']}x{m['height']} at ({m['x']}, {m['y']})")
    print("-" * 60)
    target = get_target_position()
    print(f"Target position for windows: ({target[0]}, {target[1]})")


def main():
    parser = argparse.ArgumentParser(
        description="Monitor and move windows to second monitor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python window_monitor.py          # Start monitoring
  python window_monitor.py &        # Start in background
  python window_monitor.py --list   # List current windows
  python window_monitor.py --detect # Show monitor info
  
To add new window patterns, edit WINDOW_PATTERNS at the top of this script.
        """
    )
    parser.add_argument('--list', action='store_true', help='List current windows')
    parser.add_argument('--detect', action='store_true', help='Show monitor information')
    parser.add_argument('--once', action='store_true', help='Move matching windows once and exit')
    parser.add_argument('--interval', type=float, default=POLL_INTERVAL, 
                        help=f'Polling interval in seconds (default: {POLL_INTERVAL})')
    
    args = parser.parse_args()
    
    if args.list:
        list_windows()
        return
        
    if args.detect:
        show_monitors()
        return
    
    # Get target monitor info
    target_monitor = get_target_monitor()
    
    # Create monitor
    monitor = WindowMonitor(WINDOW_PATTERNS, target_monitor)
    
    if args.once:
        count = monitor.check_and_move_windows()
        print(f"Moved {count} windows")
    else:
        monitor.run(args.interval)


if __name__ == "__main__":
    main()

