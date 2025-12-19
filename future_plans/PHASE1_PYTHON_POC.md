# Phase 1: Python Proof of Concept

## Timeline: 3-5 weeks

## Objective

Validate the hybrid control architecture using existing Python infrastructure before committing to a full ROS2 port.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PYTHON POC ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    xrobot_teleop_inspire.py                          │    │
│  │                                                                      │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │    │
│  │  │ XRobot      │  │ GMR Motion  │  │ State       │  │ Hand       │  │    │
│  │  │ Streamer    │  │ Retarget    │  │ Machine     │  │ Controller │  │    │
│  │  │             │  │             │  │             │  │            │  │    │
│  │  │ Body track  │  │ IK (MINK)   │  │ Teleop FSM  │  │ Inspire    │  │    │
│  │  │ Controller  │  │ Pose→Joint  │  │ Button cmds │  │ Modbus TCP │  │    │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └─────┬──────┘  │    │
│  │         │                │                │               │         │    │
│  └─────────┼────────────────┼────────────────┼───────────────┼─────────┘    │
│            │                │                │               │              │
│            ▼                ▼                ▼               ▼              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                            REDIS                                     │    │
│  │                                                                      │    │
│  │  mimic_obs (35-dim)  │  loco_velocity_cmd  │  hand_poses  │  state  │    │
│  └──────────┬───────────┴──────────┬──────────┴───────┬──────┴────┬────┘    │
│             │                      │                  │           │         │
│             ▼                      ▼                  ▼           ▼         │
│  ┌──────────────────┐   ┌──────────────────┐   ┌──────────┐  ┌─────────┐   │
│  │ server_low_level │   │ C++ LocoClient   │   │ Inspire  │  │ MuJoCo  │   │
│  │ _g1_real.py      │   │ Redis Bridge     │   │ Hands    │  │ Viz     │   │
│  │                  │   │                  │   │          │  │         │   │
│  │ ONNX Policy      │   │ SetVelocity()    │   │ Hardware │  │ Debug   │   │
│  │ Arm+Neck joints  │   │ SetSpeedMode()   │   │          │  │         │   │
│  └────────┬─────────┘   └────────┬─────────┘   └────┬─────┘  └─────────┘   │
│           │                      │                  │                      │
│           └──────────────────────┼──────────────────┘                      │
│                                  │                                         │
│                                  ▼                                         │
│                        ┌─────────────────┐                                 │
│                        │    G1 ROBOT     │                                 │
│                        │                 │                                 │
│                        │ Arms: rt/arm_sdk│                                 │
│                        │ Legs: LocoClient│                                 │
│                        │ Hands: Modbus   │                                 │
│                        └─────────────────┘                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Tasks

### Week 1: Joystick Locomotion

#### Task 1.1: Modify xrobot_teleop_inspire.py
- [ ] Add locomotion mode flag (tracking vs joystick)
- [ ] Extract joystick values from controller data
- [ ] Publish velocity commands to Redis
- [ ] Handle mode switching (A button toggle)

```python
# Key changes to StateMachine class
class StateMachine:
    def __init__(self):
        self.locomotion_mode = "tracking"  # or "joystick"
        self.loco_vx = 0.0
        self.loco_vy = 0.0
        self.loco_vyaw = 0.0
        
    def toggle_locomotion_mode(self):
        if self.locomotion_mode == "tracking":
            self.locomotion_mode = "joystick"
        else:
            self.locomotion_mode = "tracking"
```

#### Task 1.2: Create C++ LocoClient Bridge
- [ ] Create `g1_loco_redis_bridge.cpp`
- [ ] Implement Redis subscription (hiredis)
- [ ] Parse velocity commands from JSON
- [ ] Call LocoClient.SetVelocity()
- [ ] Handle speed mode changes

```cpp
// g1_loco_redis_bridge.cpp
void processVelocityCommand(const std::string& json_cmd) {
    auto cmd = parseJson(json_cmd);
    client.SetVelocity(cmd.vx, cmd.vy, cmd.vyaw, 0.5);
    if (cmd.speed_mode != current_speed_mode) {
        client.SetSpeedMode(cmd.speed_mode);
        current_speed_mode = cmd.speed_mode;
    }
}
```

#### Task 1.3: Integration Testing
- [ ] Test in MuJoCo simulation
- [ ] Verify velocity command flow
- [ ] Test mode switching
- [ ] Validate concurrent arm tracking

---

### Week 2: Sprint Mode (B Button)

#### Task 2.1: Button Detection
- [ ] Map B button (RightController.key_two)
- [ ] Implement hold-to-sprint logic
- [ ] Publish speed_mode to Redis

```python
def process_sprint_button(self, controller_data):
    b_pressed = controller_data.get('RightController', {}).get('key_two', False)
    
    if b_pressed and not self.is_sprinting:
        self.is_sprinting = True
        self.set_speed_mode(2)  # High speed
    elif not b_pressed and self.is_sprinting:
        self.is_sprinting = False
        self.set_speed_mode(0)  # Normal speed
```

#### Task 2.2: C++ Bridge Update
- [ ] Handle SetSpeedMode(2) for sprint
- [ ] Smooth transition between modes
- [ ] Safety checks for rapid mode changes

---

### Week 3: Jump (Y Button) & Hand Control

#### Task 3.1: Jump Implementation
**Option A: Motion File Playback**
- [ ] Download/create jump motion file
- [ ] Implement motion file loading
- [ ] Blend into current pose
- [ ] Handle recovery after jump

**Option B: If SDK Jump API Available**
- [ ] Investigate LocoClient jump capability
- [ ] Implement jump trigger

#### Task 3.2: Hand Control Improvement
- [ ] Modify close sequence in inspire_hand_wrapper.py
- [ ] Add thumb rotation at finger position 500
- [ ] Test grasp quality

```python
# Modified close sequence
def close_hand_improved(self, hand_id):
    # Close fingers first
    for pos in range(2000, 500, -100):
        self.set_finger_positions(hand_id, pos)
        time.sleep(0.02)
    
    # Rotate thumb inward at position 500
    self.set_thumb_rotation(hand_id, 500)  # Rotate thumb
    
    # Continue closing
    for pos in range(500, 0, -100):
        self.set_finger_positions(hand_id, pos)
        time.sleep(0.02)
```

---

### Week 4: Testing & Refinement

#### Task 4.1: Full System Testing
- [ ] End-to-end teleoperation test
- [ ] Locomotion + tracking concurrent test
- [ ] Button response time measurement
- [ ] Latency profiling

#### Task 4.2: Safety Implementation
- [ ] Emergency stop on controller disconnect
- [ ] Velocity limits
- [ ] Mode transition safety

---

## File Changes Summary

| File | Changes |
|------|---------|
| `xrobot_teleop_inspire.py` | Add locomotion mode, joystick processing, Redis publishing |
| `inspire_hand_wrapper.py` | Improved close sequence with thumb rotation |
| `test_joystick_locomotion.py` | Standalone joystick testing (created) |
| `g1_loco_redis_bridge.cpp` | New C++ bridge for LocoClient (to create) |

---

## Dependencies

### Python
- Existing: redis, numpy, mujoco, xrobotoolkit_sdk
- No new dependencies

### C++ (for LocoClient bridge)
- unitree_sdk2
- hiredis
- nlohmann/json (optional, for JSON parsing)

---

## Success Criteria

1. ✅ Joystick controls robot locomotion independently of arm tracking
2. ✅ Holding B button increases movement speed
3. ✅ Y button triggers jump (if feasible)
4. ✅ Hand grasp includes thumb rotation
5. ✅ System latency < 50ms local, < 250ms intercontinental
6. ✅ No degradation of existing arm tracking quality

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| LocoClient conflicts with arm_sdk | High | Test DDS topic independence |
| Jump API unavailable | Medium | Fall back to motion files |
| Latency too high | Medium | Implement prediction buffer |
| Controller disconnect | High | Implement watchdog timeout |

