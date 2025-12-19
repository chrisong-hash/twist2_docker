# Phase 2: ROS2 Port

## Timeline: 12-18 weeks

## Objective

Port the validated Python POC to a production-ready ROS2 system with:
- Better modularity and maintainability
- Industry-standard IK (MoveIt2)
- Optimized for intercontinental deployment
- Scalable architecture for future enhancements

---

## Why ROS2?

### Benefits

| Aspect | Current Python | ROS2 |
|--------|----------------|------|
| **IK Solver** | MINK (Python, custom) | MoveIt2/TRAC-IK (optimized, production-ready) |
| **Communication** | Redis (custom) | DDS (native, same as Unitree SDK) |
| **Modularity** | Monolithic scripts | Independent nodes |
| **Debugging** | Print statements | RViz2, rqt, ros2 topic echo |
| **Recording** | Custom | rosbag2 (standard) |
| **Latency** | ~15-30ms local | ~5-15ms local |

### ROS2 + Unitree SDK Compatibility

Unitree SDK2 uses CycloneDDS - the **same DDS implementation as ROS2 Humble/Iron**. This means:
- Native message compatibility
- No protocol translation needed
- Can mix ROS2 nodes with Unitree SDK code

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ROS2 NODE ARCHITECTURE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐          │
│  │ pico_tracking   │    │ motion_retarget │    │ arm_ik_node     │          │
│  │ _node           │───▶│ _node           │───▶│ (MoveIt2)       │          │
│  │                 │    │                 │    │                 │          │
│  │ xrobotoolkit    │    │ GMR logic       │    │ TRAC-IK/KDL     │          │
│  │ C++ wrapper     │    │ C++ port        │    │ Collision aware │          │
│  └─────────────────┘    └─────────────────┘    └────────┬────────┘          │
│          │                                              │                    │
│          │              /body_tracking                  │ /arm_joint_cmd    │
│          │              /controller_state               │                    │
│          ▼                                              ▼                    │
│  ┌─────────────────┐                         ┌─────────────────┐            │
│  │ locomotion_node │                         │ robot_interface │            │
│  │                 │                         │ _node           │            │
│  │ LocoClient      │────────────────────────▶│                 │            │
│  │ wrapper         │  /cmd_vel               │ rt/arm_sdk      │            │
│  │                 │  /speed_mode            │ LocoClient      │            │
│  └─────────────────┘                         └────────┬────────┘            │
│          ▲                                            │                     │
│          │                                            ▼                     │
│  ┌─────────────────┐                         ┌─────────────────┐            │
│  │ hand_control    │                         │    G1 ROBOT     │            │
│  │ _node           │────────────────────────▶│                 │            │
│  │                 │  Modbus TCP             │                 │            │
│  │ Inspire wrapper │                         │                 │            │
│  └─────────────────┘                         └─────────────────┘            │
│                                                                              │
│  ┌─────────────────┐    ┌─────────────────┐                                 │
│  │ state_machine   │    │ network_bridge  │  (For intercontinental)        │
│  │ _node           │    │ _node           │                                 │
│  │                 │    │                 │                                 │
│  │ FSM logic       │    │ WebRTC/QUIC     │                                 │
│  │ Safety checks   │    │ Compression     │                                 │
│  └─────────────────┘    └─────────────────┘                                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Tasks

### Phase 2.1: Foundation (Weeks 1-3)

#### Task 2.1.1: ROS2 Workspace Setup
- [ ] Create colcon workspace
- [ ] Set up package structure
- [ ] Configure CMakeLists.txt for Unitree SDK integration
- [ ] Create launch files

```bash
# Workspace structure
g1_teleop_ws/
├── src/
│   ├── g1_teleop_bringup/      # Launch files, configs
│   ├── g1_teleop_msgs/         # Custom message definitions
│   ├── pico_tracking/          # Body tracking node
│   ├── motion_retarget/        # GMR logic
│   ├── g1_locomotion/          # LocoClient wrapper
│   ├── g1_arm_control/         # MoveIt2 config + node
│   ├── inspire_hand_control/   # Hand control node
│   └── g1_robot_interface/     # Low-level robot interface
└── install/
```

#### Task 2.1.2: Message Definitions
- [ ] Define BodyTracking.msg
- [ ] Define ControllerState.msg
- [ ] Define LocomotionCommand.msg
- [ ] Define HandCommand.msg

```
# g1_teleop_msgs/msg/ControllerState.msg
std_msgs/Header header
bool a_button
bool b_button
bool x_button
bool y_button
float32 left_stick_x
float32 left_stick_y
float32 right_stick_x
float32 right_stick_y
float32 left_trigger
float32 right_trigger
```

#### Task 2.1.3: LocoClient ROS2 Node
- [ ] Wrap LocoClient in ROS2 node
- [ ] Subscribe to /cmd_vel (geometry_msgs/Twist)
- [ ] Implement SetSpeedMode service
- [ ] Add safety watchdog

```cpp
// g1_locomotion/src/locomotion_node.cpp
class LocomotionNode : public rclcpp::Node {
public:
    LocomotionNode() : Node("g1_locomotion") {
        cmd_vel_sub_ = create_subscription<geometry_msgs::msg::Twist>(
            "/cmd_vel", 10, 
            std::bind(&LocomotionNode::cmdVelCallback, this, _1));
        
        client_.Init();
        client_.SetFsmId(500);  // Start locomotion
    }
    
private:
    void cmdVelCallback(const geometry_msgs::msg::Twist::SharedPtr msg) {
        client_.SetVelocity(msg->linear.x, msg->linear.y, 
                           msg->angular.z, control_dt_);
    }
    
    unitree::robot::g1::LocoClient client_;
};
```

---

### Phase 2.2: Motion Retargeting (Weeks 4-7)

#### Task 2.2.1: GMR C++ Port
- [ ] Port rotation utilities (rot_utils.py → rot_utils.hpp)
- [ ] Port kinematics model (kinematics_model.py → kinematics_model.hpp)
- [ ] Port motion retarget core (motion_retarget.py → motion_retarget.hpp)
- [ ] Use Pinocchio for kinematics (replaces MINK)

**Complexity Note:** This is the most complex porting task. Consider:
- Using Pinocchio C++ directly (similar API to MINK)
- Keeping Python node with ROS2 bridge (hybrid approach)

#### Task 2.2.2: Pico Tracking Node
- [ ] Investigate xrobotoolkit C++ bindings
- [ ] Create ROS2 node wrapper
- [ ] Publish BodyTracking messages
- [ ] Publish ControllerState messages

**Risk:** xrobotoolkit may only have Python bindings. Mitigation:
- Create Python ROS2 node for tracking
- C++ for compute-heavy tasks

---

### Phase 2.3: MoveIt2 Integration (Weeks 8-10)

#### Task 2.3.1: G1 MoveIt2 Configuration
- [ ] Create URDF/SRDF for G1 arms
- [ ] Configure MoveIt2 for 7-DOF arms
- [ ] Set up collision geometry
- [ ] Tune IK parameters

```yaml
# g1_arm_control/config/kinematics.yaml
arm_left:
  kinematics_solver: trac_ik_kinematics_plugin/TRAC_IKKinematicsPlugin
  kinematics_solver_search_resolution: 0.005
  kinematics_solver_timeout: 0.005  # 5ms timeout for real-time
  
arm_right:
  kinematics_solver: trac_ik_kinematics_plugin/TRAC_IKKinematicsPlugin
  kinematics_solver_search_resolution: 0.005
  kinematics_solver_timeout: 0.005
```

#### Task 2.3.2: Real-time Servo Integration
- [ ] Configure MoveIt Servo
- [ ] Set up Cartesian streaming
- [ ] Integrate with motion retarget output

---

### Phase 2.4: Integration & Testing (Weeks 11-14)

#### Task 2.4.1: System Integration
- [ ] Create main launch file
- [ ] Configure node communication
- [ ] Set up parameter files
- [ ] Implement state machine node

#### Task 2.4.2: Testing
- [ ] Unit tests for each node
- [ ] Integration tests
- [ ] Latency benchmarking
- [ ] Real robot testing

---

### Phase 2.5: Network Optimization (Weeks 15-18)

#### Task 2.5.1: Intercontinental Bridge
- [ ] Implement network bridge node
- [ ] Add compression (pose data)
- [ ] Implement jitter buffer
- [ ] Add motion prediction

See: [INTERCONTINENTAL_ARCHITECTURE.md](./INTERCONTINENTAL_ARCHITECTURE.md)

---

## Dependencies

### ROS2 Packages
```xml
<!-- package.xml dependencies -->
<depend>rclcpp</depend>
<depend>geometry_msgs</depend>
<depend>sensor_msgs</depend>
<depend>std_msgs</depend>
<depend>moveit_ros_planning_interface</depend>
<depend>moveit_servo</depend>
<depend>tf2_ros</depend>
```

### External Libraries
- Unitree SDK2 (existing)
- Pinocchio (for kinematics)
- hiredis (optional, for Redis compatibility during transition)

### Hardware
- G1 Robot with 7-DOF arms
- Inspire hands
- Pico VR headset with body tracking

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| xrobotoolkit C++ unavailable | Medium | High | Keep Python node for tracking |
| GMR port complexity | High | Medium | Consider hybrid Python/C++ |
| MoveIt2 latency | Low | Medium | Use Servo mode, tune parameters |
| Unitree SDK conflicts | Low | High | Thorough DDS topic isolation |
| Timeline overrun | Medium | Medium | Prioritize core features |

---

## Success Criteria

1. ✅ All nodes running as independent ROS2 processes
2. ✅ End-to-end latency < 15ms (local)
3. ✅ MoveIt2 IK solving in < 5ms
4. ✅ Locomotion independent of arm control
5. ✅ rosbag2 recording capability
6. ✅ RViz2 visualization working
7. ✅ Intercontinental latency < 200ms perceived

---

## Migration Strategy

### Gradual Migration (Recommended)

```
Week 1-2:   ROS2 workspace + LocoClient node only
Week 3-4:   Add hand control node
Week 5-7:   Add MoveIt2 arm control (parallel to Python GMR)
Week 8-10:  Port GMR or validate hybrid approach
Week 11-14: Integration testing
Week 15-18: Network optimization
```

### Rollback Plan

Keep Python POC running in parallel. If ROS2 migration fails:
- Python POC remains functional
- Identify specific failing component
- Address issues incrementally

