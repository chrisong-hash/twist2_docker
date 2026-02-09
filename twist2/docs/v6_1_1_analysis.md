# V6.1.1 Static Pose Analysis

## Test Setup
- **Policy**: V6.1.1 (Pure L2 loss, 17500 iterations)
- **Test**: Static pose - motion server not running, using last Redis frame
- **Expected behavior**: Robot should stand completely still
- **Observed issues**: Hand twitching, occasional steps

---

## 🚨 KEY FINDINGS

### Raw Action Statistics
```
Overall: mean=0.1125, std=0.9682
Range: [-5.8457, 5.5127]
```

**PROBLEM**: For a static pose, actions should be near 0 with low variance. Instead we see:
- **Range of ±5.8** - HUGE for a stationary target
- **Std of 0.97** - policy is oscillating wildly

### Top 5 Jittery Joints (by frame-to-frame change)
| Joint | Avg Jerk | Action Std |
|-------|----------|------------|
| L_wrist_pitch | 0.2378 | 1.1171 |
| R_wrist_roll | 0.2366 | 2.4737 |
| L_wrist_roll | 0.2246 | 2.2059 |
| L_wrist_yaw | 0.2108 | 2.0645 |
| R_wrist_yaw | 0.1669 | 0.5873 |

**WRIST JOINTS ARE THE MAIN PROBLEM** - std of 2.0-2.5 vs 0.1-0.5 for other joints

### All Leg Joints Have High Variance
```
L_hip_pitch  : +0.7008 ± 0.2319  *** HIGH VARIANCE
L_knee       : -0.6768 ± 0.2038  *** HIGH VARIANCE
L_ankle_roll : +1.2655 ± 0.5292  *** HIGH VARIANCE
R_ankle_pitch: +0.4217 ± 0.5600  *** HIGH VARIANCE
```

This explains the "occasional steps" - legs are also oscillating significantly.

---

## Root Cause Analysis

### 1. **Feedback Loop Amplification** ⚠️ MOST LIKELY
The observation includes `last_action` (29 dims), which feeds policy output back as input.

```
Frame N: raw_action varies slightly
Frame N+1: last_action = noisy value → policy sees noise → outputs more noise
Frame N+2: even worse...
```

**Evidence**: High correlation between action variance and per-joint std

### 2. **History Buffer Accumulation**
- History buffer contains 10 previous `obs_full` 
- Each `obs_full` includes `last_action`
- Errors compound: 10 frames of accumulated noise

### 3. **Future Observation Mismatch**
- When motion server stops, `future_obs` is stale (repeated current frame)
- Policy trained with REAL future, gets FAKE future
- Confusion leads to erratic output

### 4. **L2 Loss Without Regularization**
- V6.1.1 uses pure L2 loss (no KL divergence)
- No distribution constraint = unbounded action variance
- Policy learned to match actions but not stability

---

## Recommended Fixes

### IMMEDIATE (Deployment-side)

#### Fix 1: Action Smoothing (EMA)
```python
# In server_low_level_g1_sim_v6_1_1.py
self.action_smooth_alpha = 0.7  # High = more smoothing
smoothed_action = self.action_smooth_alpha * self.last_action + (1 - self.action_smooth_alpha) * raw_action
```

#### Fix 2: Wrist Action Clamping
```python
# Clamp wrist actions more aggressively
WRIST_INDICES = [18, 19, 20, 25, 26, 27]
raw_action[WRIST_INDICES] = np.clip(raw_action[WRIST_INDICES], -0.5, 0.5)
```

#### Fix 3: Increase Wrist Damping
```python
# Higher damping fights oscillation
self.kds[18:21] = 8.0  # Left wrist (was 2.0)
self.kds[25:28] = 8.0  # Right wrist (was 2.0)
```

#### Fix 4: Zero Action in Static Pose
```python
# Detect static pose (future == current repeated)
if np.allclose(future_obs[:35], future_obs[35:70], atol=0.01):
    raw_action *= 0.1  # Dampen action significantly
```

### TRAINING-SIDE (for V6.2 or V6.3)

#### Fix 5: Action Jerk Penalty
```python
class rewards:
    class scales:
        action_jerk = -0.5  # Penalize (action - 2*last_action + last_last_action)²
```

#### Fix 6: Standing Still Reward
```python
class rewards:
    class scales:
        standing_still = 2.0  # Reward when target velocity is zero
```

#### Fix 7: Add Noise to last_action in Training
```python
# During training, add noise to last_action observation
last_action_noisy = last_action + torch.randn_like(last_action) * 0.05
```

---

## Full Log Data

### Per-Joint Statistics
```
--- Per-Joint Action Statistics (mean ± std) ---
  L_hip_pitch         : +0.7008 ± 0.2319  *** HIGH VARIANCE
  L_hip_roll          : +0.2523 ± 0.2179  *** HIGH VARIANCE
  L_hip_yaw           : +0.4823 ± 0.1514  *** HIGH VARIANCE
  L_knee              : -0.6768 ± 0.2038  *** HIGH VARIANCE
  L_ankle_pitch       : +0.5975 ± 0.5140  *** HIGH VARIANCE
  L_ankle_roll        : +1.2655 ± 0.5292  *** HIGH VARIANCE
  R_hip_pitch         : +0.4667 ± 0.2011  *** HIGH VARIANCE
  R_hip_roll          : -0.2995 ± 0.1468  *** HIGH VARIANCE
  R_hip_yaw           : -0.3195 ± 0.1764  *** HIGH VARIANCE
  R_knee              : -0.5513 ± 0.2931  *** HIGH VARIANCE
  R_ankle_pitch       : +0.4217 ± 0.5600  *** HIGH VARIANCE
  R_ankle_roll        : -0.1117 ± 0.2512  *** HIGH VARIANCE
  waist_yaw           : -0.2430 ± 0.0922
  waist_roll          : +0.0527 ± 0.0647
  waist_pitch         : +0.0442 ± 0.0638
  L_shoulder_pitch    : -0.1409 ± 0.1996  *** HIGH VARIANCE
  L_shoulder_roll     : -0.2361 ± 0.1617  *** HIGH VARIANCE
  L_shoulder_yaw      : -0.3716 ± 0.1570  *** HIGH VARIANCE
  L_elbow             : +0.0328 ± 0.3495  *** HIGH VARIANCE
  L_wrist_roll        : -1.1893 ± 2.2059  *** HIGH VARIANCE
  L_wrist_pitch       : -0.2848 ± 1.1171  *** HIGH VARIANCE
  L_wrist_yaw         : +0.9104 ± 2.0645  *** HIGH VARIANCE
  R_shoulder_pitch    : -0.2753 ± 0.1530  *** HIGH VARIANCE
  R_shoulder_roll     : +0.3949 ± 0.1275  *** HIGH VARIANCE
  R_shoulder_yaw      : +0.1663 ± 0.1572  *** HIGH VARIANCE
  R_elbow             : +0.1979 ± 0.3088  *** HIGH VARIANCE
  R_wrist_roll        : +1.1690 ± 2.4737  *** HIGH VARIANCE
  R_wrist_pitch       : +0.6394 ± 0.4923  *** HIGH VARIANCE
  R_wrist_yaw         : +0.1690 ± 0.5873  *** HIGH VARIANCE
```

### Tracking Error
```
Mean: 0.3152
Max: 4.7363
```

### Action Jerk
```
Mean abs change: 0.0793
Max abs change: 2.6369
```

---

## Conclusion

**V6.1.1 has a stability problem** in static pose scenarios. The primary cause is likely:
1. **Feedback loop** via `last_action` in observations
2. **No action regularization** in pure L2 loss

**Recommended next step**: Try deployment-side action smoothing (Fix 1) first - it's the fastest to test and should significantly reduce oscillation.

---

## Log Files
- Full JSON: `/workspace/twist2/logs_v6_1_1/v6_1_1_log_20260203_074929.json`
