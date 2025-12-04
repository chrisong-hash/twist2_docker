# TWIST2 Verification Guide

This guide will help you verify all four capabilities of TWIST2 in Docker:
1. Run simulation
2. Train
3. Test
4. Run on real robot

## Step 0: Prerequisites

Run the setup verification script:

```bash
cd /home/robo/CodeSpace/twist2_docker/twist2
bash verify_setup.sh
```

Make sure all checks pass (especially IsaacGym, PyTorch, Redis, and ONNX Runtime).

---

## Step 1: Verify Simulation (Sim2Sim)

This tests the low-level controller with the pretrained checkpoint in simulation.

### 1.1 Start Redis Server (if not already running)
```bash
redis-server --daemonize yes
```

### 1.2 Start Motion Server (Terminal 1)
```bash
cd /home/robo/CodeSpace/twist2_docker/twist2
bash run_motion_server.sh
```

**Expected output:**
- A visualization window showing the motion
- Motion streaming to Redis
- Console output showing frame updates

**What to verify:**
- ✅ Window opens without errors
- ✅ Motion plays in the visualization
- ✅ No Redis connection errors

Press `Ctrl+C` to stop when ready to test the next step.

### 1.3 Start Sim2Sim Controller (Terminal 2)
In a new terminal:
```bash
cd /home/robo/CodeSpace/twist2_docker/twist2
bash sim2sim.sh
```

**Expected output:**
- Isaac Gym window opens
- Robot stands in simulation
- FPS metrics displayed in console
- Average Policy FPS should be close to 50 Hz

**What to verify:**
- ✅ Simulation window opens
- ✅ Robot spawns and stands without falling
- ✅ Policy FPS is > 30 Hz
- ✅ No errors in console

### 1.4 Test with Motion Streaming
With sim2sim.sh still running, restart the motion server (Terminal 1):
```bash
bash run_motion_server.sh
```

**What to verify:**
- ✅ Robot in simulation follows the motion
- ✅ Robot maintains balance
- ✅ Smooth motion execution

**Success criteria:** Robot executes walking motions smoothly in simulation.

---

## Step 2: Verify Training

This tests the ability to train a new policy from scratch.

### 2.1 Check Training Data

You have two options:

**Option A: Use Example Motions (Quick Test)**
```bash
# Check if example motions exist
ls -lh assets/example_motions/
```

**Option B: Download Full Dataset (For Real Training)**
- Download TWIST2 dataset from the Google Drive link in README
- Unzip and update path in `legged_gym/motion_data_configs/twist2_dataset.yaml`

### 2.2 Start Training (Short Test)
```bash
cd /home/robo/CodeSpace/twist2_docker/twist2
bash train.sh test_train cuda:0
```

**Expected output:**
- Training starts with Isaac Gym environments
- Loss metrics printed every few iterations
- Checkpoint saves to `legged_gym/legged_gym/logs/g1_stu_future/test_train/`

**What to verify:**
- ✅ Training starts without errors
- ✅ Multiple environments spawn (you should see robot clones)
- ✅ Loss values update each iteration
- ✅ GPU is being utilized (check with `nvidia-smi`)

**Note:** You can stop training after ~100 iterations to verify it works (Ctrl+C). Full training takes many hours.

### 2.3 Export Trained Model to ONNX (Optional)

After training (even partially), export the checkpoint:
```bash
# Find your checkpoint
CKPT_PATH=$(ls legged_gym/legged_gym/logs/g1_stu_future/test_train/model_*.pt | tail -1)
echo "Found checkpoint: $CKPT_PATH"

# Export to ONNX
bash to_onnx.sh $CKPT_PATH
```

**Success criteria:** Training runs without crashes and saves checkpoints.

---

## Step 3: Verify Testing/Evaluation

This tests policy evaluation on specific motions.

### 3.1 Evaluate Pretrained Checkpoint

First, let's find an existing checkpoint to evaluate:
```bash
cd /home/robo/CodeSpace/twist2_docker/twist2

# Option 1: Use a pretrained checkpoint if available
# You'll need to locate an experiment folder with checkpoints

# Option 2: Use the checkpoint from your test training
EXPTID="test_train"  # or whatever you named it in Step 2
```

### 3.2 Modify eval.sh for Your Setup

Edit `eval.sh` to use an example motion:
```bash
#!/bin/bash
# Usage: bash eval.sh <exptid> <device>

motion_file="$(pwd)/assets/example_motions/0807_yanjie_walk_001.pkl"

task_name="g1_stu_future"
proj_name="g1_stu_future"
exptid=$1
device=$2

cd legged_gym/legged_gym/scripts

echo "Evaluating student policy with future motion support..."
echo "Task: ${task_name}"
echo "Project: ${proj_name}"
echo "Experiment ID: ${exptid}"
echo "Motion file: ${motion_file}"
echo ""

python play.py --task "${task_name}" \
               --proj_name "${proj_name}" \
               --teacher_exptid "None" \
               --exptid "${exptid}" \
               --num_envs 1 \
               --record_video \
               --device "${device}" \
               --env.motion.motion_file "${motion_file}"
```

### 3.3 Run Evaluation
```bash
bash eval.sh test_train cuda:0
```

**Expected output:**
- Single robot in Isaac Gym
- Robot follows the specified motion
- Video saved to logs directory

**What to verify:**
- ✅ Evaluation runs without errors
- ✅ Robot tracks the reference motion
- ✅ Video file is created

**Success criteria:** Policy evaluation completes and generates video output.

---

## Step 4: Verify Real Robot Deployment

⚠️ **WARNING:** Only proceed if you have a physical Unitree G1 robot and understand the safety risks!

### 4.1 Prerequisites for Real Robot

- [ ] Physical Unitree G1 robot
- [ ] Ethernet cable connection to robot
- [ ] Network configured: IP `192.168.123.222`, netmask `255.255.255.0`
- [ ] Unitree SDK Python bindings installed
- [ ] Safe testing area cleared

### 4.2 Network Setup

```bash
# Configure network interface (replace eno1 with your interface name)
sudo ip addr add 192.168.123.222/24 dev eno1

# Verify connection
ping 192.168.123.164  # Robot's IP
```

### 4.3 Put Robot in Dev Mode

Using the physical remote control:
1. Press `L2 + R2` simultaneously
2. Robot should enter dev mode (joints in damping state)

### 4.4 Update sim2real.sh

Edit `sim2real.sh` to match your network interface:
```bash
net=eno1  # Change to your interface name
```

### 4.5 Test on Real Robot

**CAUTION:** Start with robot elevated or supported!

```bash
cd /home/robo/CodeSpace/twist2_docker/twist2
bash sim2real.sh
```

Then in another terminal, start motion streaming:
```bash
bash run_motion_server.sh
```

**What to verify:**
- ✅ Robot receives commands
- ✅ Robot maintains standing position
- ✅ Robot follows commanded motions
- ✅ No sudden jerky movements

**Success criteria:** Robot executes commands safely and smoothly.

---

## Quick Verification Checklist

After completing all steps, you should have verified:

- [ ] ✅ Simulation works (sim2sim with pretrained model)
- [ ] ✅ Training starts and runs (even briefly)
- [ ] ✅ Evaluation generates video output
- [ ] ✅ Real robot connection (if hardware available)

---

## Troubleshooting

### Issue: "Redis connection refused"
**Solution:**
```bash
redis-server --daemonize yes
```

### Issue: "CUDA out of memory"
**Solution:** Reduce num_envs in training config or use smaller batch size

### Issue: "Policy FPS too low (< 30 Hz)"
**Solution:** 
- Use more powerful GPU
- Close other GPU applications
- Reduce policy_frequency in sim2sim.sh

### Issue: "Isaac Gym window doesn't open"
**Solution:**
- Check DISPLAY variable: `echo $DISPLAY`
- Ensure you're running in graphical environment
- Check Docker X11 forwarding if in container

### Issue: "Motion files not found"
**Solution:** Verify paths are absolute or relative to correct directory

---

## Using the GUI (Alternative Method)

You mentioned `gui.sh` works. The GUI provides a convenient interface:

```bash
bash gui.sh
```

From the GUI you can:
1. Run sim2sim controller
2. Run sim2real controller  
3. Start motion server
4. Start teleop mode
5. Data collection
6. Neck controller
7. ZED streaming

This is useful for quick testing without managing multiple terminals.

---

## Next Steps

After verification:
1. Train your own policy with full dataset
2. Experiment with different motions
3. Test teleoperation with PICO headset (if available)
4. Deploy to physical robot for real-world testing
5. Collect your own motion data

Good luck! 🚀

