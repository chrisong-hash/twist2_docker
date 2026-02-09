"""
BFM-Zero Logging System

Creates comparable logs between sim and real modes.
Each log entry has: timestamp, iteration, component, direction (IN/OUT), data

Log files:
- bfm_unified_{mode}_{timestamp}.log  - Main controller log
- bfm_teleop_{timestamp}.log          - Teleop server log
- bfm_motion_{timestamp}.log          - Motion server log

Format:
[TIMESTAMP] [ITER] [COMPONENT] [IN/OUT] key=value key=value ...

Example:
[0.000] [0] [SETTINGS] model_path=./model/checkpoint/model z_dim=256 ...
[0.020] [1] [MOTION_IN] frame=0 root_pos=[0.0,0.0,0.8] root_quat=[1,0,0,0] ...
[0.020] [1] [Z_OUT] z_norm=16.0 z_mode=bfm_backward
[0.020] [1] [ACTION_OUT] action=[0.1,0.2,...] target_pos=[0.0,0.1,...]
"""

import os
import time
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import json


class BFMLogger:
    """Unified logging for BFM-Zero components."""
    
    def __init__(self, component: str, mode: str = "sim", log_dir: str = "./logs"):
        """
        Initialize logger.
        
        Args:
            component: "unified", "teleop", or "motion"
            mode: "sim" or "real"
            log_dir: Directory for log files
        """
        self.component = component
        self.mode = mode
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if component == "unified":
            filename = f"bfm_unified_{mode}_{timestamp}.log"
        else:
            filename = f"bfm_{component}_{timestamp}.log"
        
        self.log_path = self.log_dir / filename
        self.log_file = open(self.log_path, 'w')
        
        # Timing
        self.start_time = time.time()
        self.iteration = 0
        
        # Write header
        self._write_header()
        
        print(f"[BFMLogger] Logging to: {self.log_path}")
    
    def _write_header(self):
        """Write log file header."""
        self.log_file.write(f"# BFM-Zero Log\n")
        self.log_file.write(f"# Component: {self.component}\n")
        self.log_file.write(f"# Mode: {self.mode}\n")
        self.log_file.write(f"# Started: {datetime.now().isoformat()}\n")
        self.log_file.write(f"# Format: [TIME_SEC] [ITER] [TAG] key=value ...\n")
        self.log_file.write(f"#\n")
        self.log_file.flush()
    
    def _format_value(self, v: Any) -> str:
        """Format a value for logging."""
        if isinstance(v, np.ndarray):
            if v.size <= 30:
                return f"[{','.join(f'{x:.4f}' for x in v.flatten())}]"
            else:
                # For large arrays, show first/last few and stats
                flat = v.flatten()
                return f"[{flat[0]:.4f},...,{flat[-1]:.4f}](len={len(flat)},mean={flat.mean():.4f},std={flat.std():.4f})"
        elif isinstance(v, (list, tuple)):
            if len(v) <= 30:
                return f"[{','.join(f'{x:.4f}' if isinstance(x, float) else str(x) for x in v)}]"
            else:
                return f"[{v[0]},...,{v[-1]}](len={len(v)})"
        elif isinstance(v, float):
            return f"{v:.6f}"
        elif isinstance(v, dict):
            # For nested dicts, just show keys
            return f"{{{','.join(v.keys())}}}"
        else:
            return str(v)
    
    def _format_entry(self, tag: str, **kwargs) -> str:
        """Format a log entry."""
        elapsed = time.time() - self.start_time
        parts = [f"[{elapsed:.3f}]", f"[{self.iteration}]", f"[{tag}]"]
        
        for k, v in kwargs.items():
            parts.append(f"{k}={self._format_value(v)}")
        
        return " ".join(parts)
    
    def log(self, tag: str, **kwargs):
        """Log an entry."""
        entry = self._format_entry(tag, **kwargs)
        self.log_file.write(entry + "\n")
        self.log_file.flush()
    
    def log_settings(self, settings: Dict[str, Any]):
        """Log settings at startup."""
        self.log("SETTINGS", **settings)
    
    def log_motion_in(self, frame: int, root_pos: Any, root_quat: Any, 
                      root_vel: Any = None, root_ang_vel: Any = None,
                      dof_pos: Any = None, **kwargs):
        """Log incoming motion data."""
        data = {
            "frame": frame,
            "root_pos": root_pos,
            "root_quat": root_quat,
        }
        if root_vel is not None:
            data["root_vel"] = root_vel
        if root_ang_vel is not None:
            data["root_ang_vel"] = root_ang_vel
        if dof_pos is not None:
            data["dof_pos"] = dof_pos
        data.update(kwargs)
        self.log("MOTION_IN", **data)
    
    def log_obs_in(self, obs_type: str, **kwargs):
        """Log observation input."""
        self.log(f"OBS_{obs_type}_IN", **kwargs)
    
    def log_z_out(self, z_norm: float, z_mode: str, z_sample: Any = None, **kwargs):
        """Log z computation output."""
        data = {"z_norm": z_norm, "z_mode": z_mode}
        if z_sample is not None:
            data["z_sample"] = z_sample[:5] if len(z_sample) > 5 else z_sample
        data.update(kwargs)
        self.log("Z_OUT", **data)
    
    def log_action_out(self, action: Any, target_pos: Any = None, **kwargs):
        """Log action output."""
        data = {"action": action}
        if target_pos is not None:
            data["target_pos"] = target_pos
        data.update(kwargs)
        self.log("ACTION_OUT", **data)
    
    def log_robot_state(self, dof_pos: Any, dof_vel: Any = None, 
                        ang_vel: Any = None, quat: Any = None, **kwargs):
        """Log robot state."""
        data = {"dof_pos": dof_pos}
        if dof_vel is not None:
            data["dof_vel"] = dof_vel
        if ang_vel is not None:
            data["ang_vel"] = ang_vel
        if quat is not None:
            data["quat"] = quat
        data.update(kwargs)
        self.log("ROBOT_STATE", **data)
    
    def log_state_machine(self, state: str, prev_state: str = None, **kwargs):
        """Log state machine transition."""
        data = {"state": state}
        if prev_state is not None:
            data["prev_state"] = prev_state
        data.update(kwargs)
        self.log("STATE", **data)
    
    def next_iteration(self):
        """Increment iteration counter."""
        self.iteration += 1
    
    def close(self):
        """Close log file."""
        self.log("END", total_iterations=self.iteration, 
                 total_time=time.time() - self.start_time)
        self.log_file.close()
        print(f"[BFMLogger] Closed: {self.log_path}")


    def log_arm_joints(self, dof_pos, action=None, target_pos=None):
        """Log arm joint data specifically (joints 15-28)."""
        import numpy as np
        if isinstance(dof_pos, np.ndarray) and len(dof_pos) >= 29:
            left_arm = dof_pos[15:22]
            right_arm = dof_pos[22:29]
        else:
            left_arm = [0]*7
            right_arm = [0]*7
        
        data = {
            "L_sh_p": float(left_arm[0]),
            "L_sh_r": float(left_arm[1]),
            "L_sh_y": float(left_arm[2]),
            "L_elb": float(left_arm[3]),
            "R_sh_p": float(right_arm[0]),
            "R_sh_r": float(right_arm[1]),
            "R_sh_y": float(right_arm[2]),
            "R_elb": float(right_arm[3]),
        }
        
        if action is not None and hasattr(action, '__len__') and len(action) >= 29:
            data["L_sh_p_act"] = float(action[15])
            data["L_elb_act"] = float(action[18])
            data["R_sh_p_act"] = float(action[22])
            data["R_elb_act"] = float(action[25])
        
        if target_pos is not None and hasattr(target_pos, '__len__') and len(target_pos) >= 29:
            data["L_sh_p_tgt"] = float(target_pos[15])
            data["L_elb_tgt"] = float(target_pos[18])
            data["R_sh_p_tgt"] = float(target_pos[22])
            data["R_elb_tgt"] = float(target_pos[25])
        
        self.log("ARM_JOINTS", **data)


def compare_logs(log1_path: str, log2_path: str, output_path: str = None):
    """
    Compare two log files and show differences.
    
    Args:
        log1_path: Path to first log (e.g., sim)
        log2_path: Path to second log (e.g., real)
        output_path: Optional path for diff output
    """
    import re
    
    def parse_log(path):
        entries = []
        with open(path, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                # Parse: [TIME] [ITER] [TAG] key=value ...
                match = re.match(r'\[(\d+\.\d+)\]\s+\[(\d+)\]\s+\[(\w+)\]\s+(.*)', line.strip())
                if match:
                    time_sec, iteration, tag, data_str = match.groups()
                    # Parse key=value pairs
                    data = {}
                    for kv in re.findall(r'(\w+)=([^\s]+(?:\s+(?!\w+=)[^\s]+)*)', data_str):
                        data[kv[0]] = kv[1]
                    entries.append({
                        'time': float(time_sec),
                        'iter': int(iteration),
                        'tag': tag,
                        'data': data
                    })
        return entries
    
    log1 = parse_log(log1_path)
    log2 = parse_log(log2_path)
    
    # Group by iteration and tag
    def group_entries(entries):
        grouped = {}
        for e in entries:
            key = (e['iter'], e['tag'])
            grouped[key] = e
        return grouped
    
    g1 = group_entries(log1)
    g2 = group_entries(log2)
    
    # Find differences
    all_keys = set(g1.keys()) | set(g2.keys())
    diffs = []
    
    for key in sorted(all_keys):
        iter_num, tag = key
        e1 = g1.get(key)
        e2 = g2.get(key)
        
        if e1 is None:
            diffs.append(f"[{iter_num}] [{tag}] ONLY IN LOG2")
        elif e2 is None:
            diffs.append(f"[{iter_num}] [{tag}] ONLY IN LOG1")
        else:
            # Compare data
            all_data_keys = set(e1['data'].keys()) | set(e2['data'].keys())
            for dk in all_data_keys:
                v1 = e1['data'].get(dk, 'N/A')
                v2 = e2['data'].get(dk, 'N/A')
                if v1 != v2:
                    diffs.append(f"[{iter_num}] [{tag}] {dk}: {v1} vs {v2}")
    
    # Output
    output = f"# Log Comparison\n"
    output += f"# Log1: {log1_path}\n"
    output += f"# Log2: {log2_path}\n"
    output += f"# Differences: {len(diffs)}\n"
    output += f"#\n"
    output += "\n".join(diffs[:1000])  # Limit output
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(output)
        print(f"Diff written to: {output_path}")
    else:
        print(output)
    
    return diffs


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        compare_logs(sys.argv[1], sys.argv[2], 
                     sys.argv[3] if len(sys.argv) > 3 else None)
    else:
        print("Usage: python bfm_logger.py log1.log log2.log [output.diff]")
