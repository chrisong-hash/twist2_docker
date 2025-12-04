# TWIST2 Docker Environment

🤖 A ready-to-use Docker environment for [TWIST2](https://github.com/amazon-far/TWIST2) humanoid robot control, training, and deployment.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ Features

- ✅ **Zero configuration** - Works out of the box
- ✅ **GPU accelerated** - CUDA support for training
- ✅ **Complete setup** - All dependencies pre-installed
- ✅ **Helper scripts** - One-command installation
- ✅ **Example motions** - Test immediately without dataset
- ✅ **Documentation** - Comprehensive guides included

## 🚀 Quick Start

### Prerequisites
- Docker with NVIDIA GPU support
- NVIDIA drivers installed
- ~15 GB disk space

### Installation (3 commands)

```bash
git clone https://github.com/YOUR_USERNAME/twist2_docker.git
cd twist2_docker
./scripts/install.sh
```

That's it! Wait ~15 minutes for the build to complete.

### Start Training

```bash
./scripts/run.sh
docker exec -it twist2 bash
cd /workspace/twist2
bash train.sh my_experiment cuda:0
```

Training starts immediately with included example motions! 🎉

## 📚 Documentation

- [Quick Start Guide](QUICK_START.md)
- [Docker Verification Steps](DOCKER_VERIFICATION_STEPS.md)
- [Training Notes](twist2/TRAINING_NOTES.md)
- [Verification Guide](twist2/VERIFICATION_GUIDE.md)
- [Setup Complete](SETUP_COMPLETE.md)

## 🎯 What's Included

### Pre-installed Software
- Isaac Gym (physics simulation)
- PyTorch with CUDA
- Redis (motion streaming)
- ONNXRuntime GPU (inference)
- MuJoCo (physics engine)
- All TWIST2 dependencies

### Pre-configured
- WandB disabled by default (no login prompts)
- Example motions ready to use
- Auto-start Redis server
- GPU-accelerated training

## 🧪 Verified Capabilities

| Feature | Status | Command |
|---------|--------|---------|
| Environment Setup | ✅ | `bash verify_docker_setup.sh` |
| Simulation | ✅ | `bash sim2sim.sh` |
| Training | ✅ | `bash train.sh <exp> cuda:0` |
| Motion Streaming | ✅ | `bash run_motion_server.sh` |
| Real Robot | ⚠️ | Requires Unitree G1 hardware |

Tested on Ubuntu 20.04 with NVIDIA RTX 4090.

## 📁 Project Structure

```
twist2_docker/
├── Dockerfile              # Docker image definition
├── docker-compose.yml      # Docker compose config
├── requirements.txt        # Python dependencies
├── scripts/                # Helper scripts
│   ├── install.sh          # First-time setup
│   ├── run.sh              # Start container
│   └── rebuild_docker.sh   # Rebuild from scratch
├── isaacgym/              # Isaac Gym (not in repo - download separately)
└── twist2/                # TWIST2 source code
```

## 🔧 Helper Scripts

| Script | Purpose |
|--------|---------|
| `scripts/install.sh` | First-time installation |
| `scripts/run.sh` | Start/resume container |
| `scripts/rebuild_docker.sh` | Clean rebuild |
| `scripts/clean_all.sh` | Remove everything |

## 📖 Usage Examples

### Quick Training Test
```bash
docker exec -it twist2 bash
cd /workspace/twist2
bash train.sh test_run cuda:0
```

### Simulation with Motion Playback
```bash
# Terminal 1
docker exec -it twist2 bash
bash sim2sim.sh

# Terminal 2 (new terminal)
docker exec -it twist2 bash
bash run_motion_server.sh
```

### Using GUI
```bash
docker exec -it twist2 bash
bash gui.sh
```

## 🔥 What's Different From Original?

This Docker environment includes:

- ✅ Complete dependency management
- ✅ Automatic configuration
- ✅ Example motions pre-configured
- ✅ No WandB login required
- ✅ Training works immediately
- ✅ Helper scripts for easy management

## 📥 Full Dataset (Optional)

The Docker includes 10 example motions for testing. For full training:

1. Download TWIST2 dataset from original repo
2. Mount in `docker-compose.yml`:
```yaml
volumes:
  - /path/to/dataset:/workspace/datasets/TWIST2_full:ro
```
3. Update config path in `twist2_dataset.yaml`

See [TRAINING_NOTES.md](twist2/TRAINING_NOTES.md) for details.

## 🐛 Troubleshooting

### "Cannot open display"
```bash
# On host
xhost +local:docker
```

### "CUDA not available"
```bash
# Check NVIDIA Docker
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi
```

### Container won't start
```bash
docker logs twist2
./scripts/rebuild_docker.sh
```

See documentation for more troubleshooting tips.

## 🙏 Acknowledgments

- **TWIST2**: [amazon-far/TWIST2](https://github.com/amazon-far/TWIST2)
- **Isaac Gym**: NVIDIA
- **Original Authors**: Yanjie Ze et al.

## 📄 License

MIT License - See [LICENSE](LICENSE) file

## 📧 Contact

For issues with this Docker environment, open a GitHub issue.

For TWIST2 questions, contact: yanjieze@stanford.edu

---

**⭐ If this helped you, please star the repo!**

Made with ❤️ for the humanoid robotics community

