# TWIST2 Docker Verification - Step by Step

This guide is specifically for running TWIST2 inside your Docker container.

## Your Current Setup

Based on your docker-compose.yml:
- Container name: `twist2`
- TWIST2 code mounted at: `/workspace/twist2` (inside container)
- IsaacGym at: `/opt/isaacgym` (inside container)
- Network mode: `host` (container shares host network)
- Display: X11 forwarded for GUI

---

## Step 0: Rebuild and Start Container (if needed)

```bash
# On HOST machine (outside Docker)
cd /home/robo/CodeSpace/twist2_docker

# Stop and remove old containers
docker compose down --remove-orphans

# Rebuild and start fresh
docker compose up -d --force-recreate

# Verify container is running
docker ps | grep twist2
```

---

## Step 1: Enter the Container

```bash
# On HOST machine
docker exec -it twist2 bash
```

You should now see a prompt like: `root@yourhostname:/workspace#`

---

## Step 2: Run Verification Script

```bash
# Inside Docker container
cd /workspace/twist2
bash verify_docker_setup.sh
```

This will check:
- ✅ Python and packages (PyTorch, IsaacGym, etc.)
- ✅ GPU/CUDA availability
- ✅ Files and assets
- ✅ Display/X11 setup
- ✅ Required modules

### Common Issues & Fixes

**Issue: Redis not installed**
```bash
apt-get update && apt-get install -y redis-server
redis-server --daemonize yes
```

**Issue: Missing Python packages**
```bash
pip install redis onnxruntime-gpu opencv-python
```

**Issue: legged_gym/rsl_rl/pose not installed**
```bash
cd /workspace/twist2/legged_gym && pip install -e .
cd /workspace/twist2/rsl_rl && pip install -e .
cd /workspace/twist2/pose && pip install -e .
```

**Issue: X11/Display errors**
```bash
# On HOST (outside Docker), run:
xhost +local:docker
```

---

## Step 3: Test Simulation (Simple Standing Test)

Once verification passes, test if simulation works:

```bash
# Inside Docker container
cd /workspace/twist2

# Start Redis if not running
redis-server --daemonize yes

# Test sim2sim (robot should stand in Isaac Gym window)
bash sim2sim.sh
```

**Expected behavior:**
- Isaac Gym window opens
- G1 robot spawns and stands
- Console shows "Policy FPS" around 30-50 Hz
- No errors in terminal

Press `Ctrl+C` to stop.

---

## Step 4: Test with Motion Playback

This tests the full pipeline: motion server → Redis → low-level controller

### Terminal 1 (Motion Server):
```bash
# Inside Docker container
cd /workspace/twist2
bash run_motion_server.sh
```

Expected: Window opens showing motion visualization

### Terminal 2 (Low-level Controller):
```bash
# In a NEW terminal on HOST:
docker exec -it twist2 bash

# Inside container:
cd /workspace/twist2
bash sim2sim.sh
```

Expected: Robot in Isaac Gym follows the walking motion

---

## Step 5: Quick Training Test (Optional)

Test if training pipeline works:

```bash
# Inside Docker container
cd /workspace/twist2

# Run training for 100 iterations (just to test, not full training)
bash train.sh test_docker cuda:0
```

Press `Ctrl+C` after ~100 iterations to stop.

**Expected:**
- Multiple robot clones appear in Isaac Gym
- Training loss values print every iteration
- Checkpoints save to `legged_gym/legged_gym/logs/`

---

## Quick Command Reference

### Start container:
```bash
# On HOST
cd /home/robo/CodeSpace/twist2_docker
docker compose up -d
```

### Enter container:
```bash
# On HOST
docker exec -it twist2 bash
```

### Stop container:
```bash
# On HOST
docker compose down
```

### Check container logs:
```bash
# On HOST
docker logs twist2
```

### Multiple terminals in same container:
```bash
# Terminal 1
docker exec -it twist2 bash

# Terminal 2 (NEW host terminal)
docker exec -it twist2 bash

# Terminal 3 (NEW host terminal)
docker exec -it twist2 bash
```

---

## Summary: 4 Capabilities Verification

| Capability | Test Command | Success Criteria |
|------------|--------------|------------------|
| **1. Simulation** | `bash sim2sim.sh` | Isaac Gym opens, robot stands |
| **2. Training** | `bash train.sh test_docker cuda:0` | Training starts, envs spawn |
| **3. Testing** | `bash eval.sh <exptid> cuda:0` | Evaluation runs, video saved |
| **4. Real Robot** | `bash sim2real.sh` | Requires physical G1 + network setup |

---

## Troubleshooting

### Container won't start
```bash
# Check logs
docker logs twist2

# Try rebuilding
docker compose build --no-cache
docker compose up -d
```

### GPU not accessible
```bash
# Inside container, check:
nvidia-smi

# If fails, check Docker runtime on HOST:
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi
```

### Display/GUI not working
```bash
# On HOST:
echo $DISPLAY
xhost +local:docker

# Inside container:
echo $DISPLAY
```

---

## Next Steps

After successful verification:
1. ✅ Simulation works → Try different motions in `assets/example_motions/`
2. ✅ Training works → Download full dataset and train your own policy
3. ✅ Use GUI → `bash gui.sh` for convenient interface
4. ✅ Real robot → Follow sim2real setup in VERIFICATION_GUIDE.md

Good luck! 🚀

