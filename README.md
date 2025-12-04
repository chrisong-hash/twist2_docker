# TWIST2 Docker Environment

A ready-to-use Docker environment for TWIST2 humanoid robot control, training, and deployment.

## ⚡ Super Quick Start

```bash
git clone <your-repo>
cd twist2_docker
chmod +x scripts/*.sh
./scripts/install.sh    # First time setup
./scripts/run.sh        # Start container
docker exec -it twist2 bash
```

---

## 🚀 Detailed Setup

### Prerequisites
- Docker with NVIDIA GPU support
- NVIDIA drivers installed on host
- X11 display for GUI (Isaac Gym visualization)

### Option A: Using Helper Scripts (Recommended)

```bash
# 1. Clone the repository
git clone <your-repo>
cd twist2_docker

# 2. Make scripts executable
chmod +x scripts/*.sh

# 3. Run installation (checks dependencies, builds image)
./scripts/install.sh

# 4. Start the container
./scripts/run.sh

# 5. Enter and verify
docker exec -it twist2 bash
cd /workspace/twist2
bash verify_docker_setup.sh
```

See [scripts/README.md](scripts/README.md) for more details on the helper scripts.

### Option B: Manual Setup

```bash
# 1. Clone and setup
git clone <your-repo>
cd twist2_docker

# 2. Allow Docker to access X11 display
xhost +local:docker

# Optional: To make it persistent (run once):
echo 'xhost +local:docker >/dev/null 2>&1' >> ~/.bashrc

# 3. Build the Docker image (first time only, ~10-15 minutes)
docker compose build

# 4. Start the container
docker compose up -d

# 5. Enter the container
docker exec -it twist2 bash
```

### 3. Verify Installation

```bash
# Inside the container
cd /workspace/twist2
bash verify_docker_setup.sh
```

All checks should pass ✅

## 📋 What's Included

### Pre-installed Software
- ✅ Isaac Gym (physics simulation)
- ✅ PyTorch with CUDA support
- ✅ Redis (for motion streaming)
- ✅ ONNXRuntime GPU (for inference)
- ✅ MuJoCo (physics engine)
- ✅ All TWIST2 dependencies
- ✅ Pretrained checkpoint (`assets/ckpts/twist2_1017_20k.onnx`)
- ✅ Example motion files

### TWIST2 Modules
- `legged_gym` - Training environments
- `rsl_rl` - Reinforcement learning
- `pose` - Pose estimation and retargeting

## 🧪 Testing the Setup

### Test 1: Simulation (Sim2Sim)

**Terminal 1 - Low-level controller:**
```bash
docker exec -it twist2 bash
cd /workspace/twist2
bash sim2sim.sh
```

Expected: Isaac Gym window opens with G1 robot standing

**Terminal 2 - Motion server:**
```bash
docker exec -it twist2 bash
cd /workspace/twist2
bash run_motion_server.sh
```

Expected: Robot in Isaac Gym follows the walking motion

### Test 2: Training

```bash
docker exec -it twist2 bash
cd /workspace/twist2

# Quick test (100 iterations)
bash train.sh test_docker cuda:0
# Press Ctrl+C after confirming it works

# Full training (requires dataset)
# Download dataset and update path in legged_gym/motion_data_configs/twist2_dataset.yaml
bash train.sh my_experiment cuda:0
```

### Test 3: GUI Interface

```bash
docker exec -it twist2 bash
cd /workspace/twist2
bash gui.sh
```

Control everything from a single interface!

### Test 4: Real Robot Deployment

⚠️ **Only if you have Unitree G1 hardware**

1. Connect robot via Ethernet
2. Configure network: IP `192.168.123.222`, netmask `255.255.255.0`
3. Put robot in dev mode (L2+R2 on remote)
4. Update network interface in `sim2real.sh`
5. Run: `bash sim2real.sh`

See [VERIFICATION_GUIDE.md](twist2/VERIFICATION_GUIDE.md) for detailed instructions.

## 📁 Directory Structure

```
twist2_docker/
├── README.md                # This file
├── requirements.txt         # Python dependencies
├── docker-compose.yml       # Docker compose configuration
├── Dockerfile               # Docker image definition
├── scripts/                 # Helper scripts
│   ├── install.sh           # First-time setup
│   ├── run.sh               # Start container
│   ├── rebuild_docker.sh    # Rebuild from scratch
│   └── README.md            # Scripts documentation
├── isaacgym/               # Isaac Gym installation
└── twist2/                 # TWIST2 source code (mounted)
    ├── assets/
    │   ├── ckpts/          # Pretrained checkpoints
    │   └── example_motions/ # Sample motion files
    ├── legged_gym/         # Training environments
    ├── rsl_rl/             # RL algorithms
    ├── pose/               # Pose estimation
    ├── deploy_real/        # Deployment scripts
    ├── sim2sim.sh          # Simulation test
    ├── train.sh            # Training script
    ├── gui.sh              # GUI interface
    └── verify_docker_setup.sh  # Setup verification
```

## 🔧 Common Commands

### Using Helper Scripts (Recommended)

```bash
# First time setup
./scripts/install.sh

# Start/resume work
./scripts/run.sh

# Rebuild everything
./scripts/rebuild_docker.sh
```

### Container Management (Manual)

```bash
# Start container
docker compose up -d

# Stop container
docker compose down

# Rebuild after Dockerfile changes
docker compose build --no-cache
docker compose up -d --force-recreate

# View container logs
docker logs twist2

# Multiple terminals
docker exec -it twist2 bash  # Terminal 1
docker exec -it twist2 bash  # Terminal 2 (new host terminal)
docker exec -it twist2 bash  # Terminal 3 (new host terminal)
```

### Inside Container

```bash
# Check Redis status
redis-cli ping  # Should return: PONG

# Check GPU
nvidia-smi

# Check Python packages
pip list | grep -E "torch|isaac|redis|onnx|mujoco"

# Run verification
bash verify_docker_setup.sh
```

## 🐛 Troubleshooting

### Issue: "Cannot open display"
**Solution:** On host, run `xhost +local:docker`

### Issue: "CUDA not available"
**Solution:** 
- Check: `docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi`
- Ensure nvidia-docker2 is installed
- Restart Docker daemon

### Issue: "Redis connection refused"
**Solution:** Redis should auto-start, but you can manually start:
```bash
redis-server --daemonize yes
```

### Issue: Container won't start
**Solution:**
```bash
docker logs twist2  # Check logs
docker compose down
docker compose up -d
```

### Issue: Low FPS in simulation
**Solution:**
- Close other GPU applications
- Reduce `num_envs` for training
- Check GPU utilization: `nvidia-smi`

## 📚 Additional Documentation

- [DOCKER_VERIFICATION_STEPS.md](DOCKER_VERIFICATION_STEPS.md) - Detailed verification guide
- [twist2/VERIFICATION_GUIDE.md](twist2/VERIFICATION_GUIDE.md) - Complete capability testing
- [TWIST2 Paper](https://yanjieze.com/TWIST2) - Research paper and project page

## 🎯 Verified Capabilities

This Docker environment has been tested and verified for:

| Capability | Status | Command |
|------------|--------|---------|
| **Simulation** | ✅ | `bash sim2sim.sh` |
| **Training** | ✅ | `bash train.sh <exp> cuda:0` |
| **Evaluation** | ✅ | `bash eval.sh <exp> cuda:0` |
| **Real Robot** | ⚠️ | `bash sim2real.sh` (requires hardware) |

## 🤝 Contributing

If you use this Docker setup:
1. Report issues via GitHub Issues
2. Share improvements via Pull Requests
3. Star the repo if it helped you! ⭐

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- TWIST2 repository: [amazon-far/TWIST2](https://github.com/amazon-far/TWIST2)
- Isaac Gym by NVIDIA
- AMASS and OMOMO datasets (research use only)

---

**Ready to build humanoid robots?** 🤖🚀

For questions: yanjieze@stanford.edu (original TWIST2 author)

