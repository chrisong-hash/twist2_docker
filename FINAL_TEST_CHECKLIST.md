# TWIST2 Docker - Final Test Checklist

Complete from-scratch verification of all fixes.

## Issues We Fixed

1. ✅ docker-compose.yml missing build config
2. ✅ Module installation order (rsl_rl → legged_gym → pose)
3. ✅ Missing OpenCV (python + system libs)
4. ✅ Missing vim/nano
5. ✅ WandB login prompts
6. ✅ Motion file paths (non-existent dataset)
7. ✅ Training config (wrong task needing teacher model)

---

## Step 1: Clean Everything

```bash
cd /home/robo/CodeSpace/twist2_docker

# Use the clean script
chmod +x scripts/*.sh
./scripts/clean_all.sh

# Verify cleanup
docker ps -a | grep twist2  # Should be empty
docker images | grep twist2  # Should be empty
```

**Expected:** All containers and images removed

---

## Step 2: Fresh Build

```bash
# Build from scratch
./scripts/install.sh
```

**Expected output:**
- ✅ Docker installed check passes
- ✅ NVIDIA Docker check passes  
- ✅ X11 setup completes
- ✅ Build starts and shows progress
- ✅ Build completes in ~10-15 minutes
- ✅ Success message displayed

**Watch for:**
- [ ] apt packages install (redis-server, vim, nano, libglib2.0-0, etc.)
- [ ] Python packages install (redis, onnxruntime-gpu, mujoco, opencv-python)
- [ ] rsl_rl installs first
- [ ] legged_gym installs second
- [ ] pose installs third
- [ ] example_motions.yaml created
- [ ] Configs updated automatically

---

## Step 3: Start Container

```bash
./scripts/run.sh
```

**Expected:**
- ✅ Container created
- ✅ Instructions to enter shown

---

## Step 4: Verify Installation

```bash
docker exec -it twist2 bash
cd /workspace/twist2
bash verify_docker_setup.sh
```

**All checks should pass:**
- [ ] ✅ Running inside Docker
- [ ] ✅ IsaacGym
- [ ] ✅ PyTorch with CUDA
- [ ] ✅ Redis (running)
- [ ] ✅ ONNXRuntime
- [ ] ✅ OpenCV
- [ ] ✅ NumPy
- [ ] ✅ MuJoCo
- [ ] ✅ GPU detected (RTX 4090)
- [ ] ✅ Pretrained checkpoint exists
- [ ] ✅ Example motions (10 files)
- [ ] ✅ All modules (legged_gym, rsl_rl, pose)
- [ ] ✅ DISPLAY set
- [ ] ✅ All scripts exist

---

## Step 5: Check WandB Configuration

```bash
# Inside container
echo $WANDB_MODE
echo $WANDB_DISABLED
```

**Expected:**
```
disabled
true
```

---

## Step 6: Check Example Motions Config

```bash
# Inside container
cat /workspace/twist2/legged_gym/motion_data_configs/example_motions.yaml | head -20
```

**Expected:**
- [ ] ✅ File exists
- [ ] ✅ root_path: /workspace/twist2/assets/example_motions
- [ ] ✅ Lists 10 motion files

```bash
# Verify config was updated
grep motion_file /workspace/twist2/legged_gym/legged_gym/envs/g1/g1_mimic_distill_config.py
```

**Expected:** Shows `example_motions.yaml`

---

## Step 7: Test Training (Quick - 10 iterations)

```bash
# Inside container
cd /workspace/twist2/legged_gym/legged_gym/scripts

# Quick test with 10 iterations
python train.py --task g1_priv_mimic \
                --proj_name test_from_scratch \
                --exptid test_$(date +%s) \
                --device cuda:0 \
                --max_iterations 10
```

**Expected:**
- [ ] ✅ No WandB login prompts
- [ ] ✅ Motion files load (10 example motions)
- [ ] ✅ Isaac Gym environments spawn
- [ ] ✅ Training starts
- [ ] ✅ GPU utilization high (check with `nvidia-smi`)
- [ ] ✅ Steps/s > 30,000
- [ ] ✅ Completes 10 iterations
- [ ] ✅ Checkpoint saves to logs/

---

## Step 8: Test with train.sh Script

```bash
# Inside container
cd /workspace/twist2
bash train.sh final_test cuda:0
```

**Expected:**
- [ ] ✅ Uses g1_priv_mimic task
- [ ] ✅ No teacher model errors
- [ ] ✅ Training starts immediately
- [ ] ✅ No manual configuration needed

**Stop after confirming it starts (Ctrl+C)**

---

## Step 9: Test Simulation

```bash
# Inside container, Terminal 1
cd /workspace/twist2
bash sim2sim.sh
```

**Expected:**
- [ ] ✅ Isaac Gym window opens
- [ ] ✅ Robot spawns
- [ ] ✅ Robot stands
- [ ] ✅ Policy FPS ~40-50 Hz

**Keep running, open Terminal 2:**

```bash
# On HOST, new terminal
docker exec -it twist2 bash
cd /workspace/twist2
bash run_motion_server.sh
```

**Expected:**
- [ ] ✅ Motion visualization window opens
- [ ] ✅ Robot in Isaac Gym follows motion
- [ ] ✅ Smooth walking behavior

---

## Step 10: Check Editors Available

```bash
# Inside container
which vim
which nano
```

**Expected:** Both found

---

## Step 11: Verify Redis Auto-starts

```bash
# Exit and re-enter container
exit
docker exec -it twist2 bash

# Check if Redis is running
redis-cli ping
```

**Expected:** `PONG`

---

## Final Checklist

### Environment Setup
- [ ] Docker builds successfully
- [ ] All dependencies installed
- [ ] No manual fixes needed

### Configuration
- [ ] WandB disabled by default
- [ ] Example motions configured
- [ ] Training uses correct task
- [ ] Redis auto-starts

### Functionality  
- [ ] Training works out-of-the-box
- [ ] Simulation works
- [ ] Motion streaming works
- [ ] GPU utilized properly

### User Experience
- [ ] Helper scripts work
- [ ] Documentation clear
- [ ] Zero manual configuration
- [ ] Editors available

---

## Expected Results Summary

| Test | Expected Time | Success Criteria |
|------|---------------|------------------|
| Clean | 1 min | All removed |
| Build | 10-15 min | Completes with success |
| Start | 5 sec | Container running |
| Verify | 30 sec | All checks pass |
| Train test | 2-3 min | 10 iterations complete |
| Simulation | 10 sec | Robot appears and moves |

---

## If Any Test Fails

Document:
1. Which step failed
2. Error message
3. What was expected vs actual

Then we can fix before final release.

---

## Success Criteria

✅ **All tests pass without ANY manual intervention**

Users should be able to:
```bash
git clone <repo>
cd twist2_docker
./scripts/install.sh
./scripts/run.sh
docker exec -it twist2 bash
cd /workspace/twist2
bash train.sh my_experiment cuda:0  # Just works!
```

**Zero configuration, zero errors, zero manual fixes!** 🎉

