# Changelog

All notable improvements to the TWIST2 Docker environment.

## [1.0.0] - Initial Complete Release

### 🎉 What's New

#### 1. Simplified Dockerfile
- ✅ Created `requirements.txt` for centralized dependency management
- ✅ Cleaner Dockerfile structure
- ✅ All dependencies pre-installed (redis, onnxruntime-gpu, mujoco)
- ✅ TWIST2 submodules auto-installed (legged_gym, rsl_rl, pose)
- ✅ Redis auto-starts on container launch

#### 2. Helper Scripts
- ✅ `scripts/install.sh` - One-command installation with checks
- ✅ `scripts/run.sh` - Smart container start/resume
- ✅ `scripts/rebuild_docker.sh` - Clean rebuild process
- ✅ `scripts/README.md` - Scripts documentation

#### 3. Complete Documentation
- ✅ `README.md` - Main user documentation with quick start
- ✅ `QUICK_START.md` - Fast setup guide
- ✅ `DOCKER_VERIFICATION_STEPS.md` - Detailed testing steps
- ✅ `DOCKER_SETUP_NOTES.md` - Technical implementation notes
- ✅ `twist2/verify_docker_setup.sh` - Automated verification script
- ✅ `twist2/VERIFICATION_GUIDE.md` - Complete capability testing

#### 4. Improved User Experience
- ✅ X11 setup with persistent option clearly explained
- ✅ Super quick start section (3 commands to get running)
- ✅ Both script-based and manual workflows documented
- ✅ Clear directory structure documentation
- ✅ Comprehensive troubleshooting guides

### 📦 What's Included

**System Packages:**
- redis-server
- All X11 and OpenGL dependencies
- Python 3.8 with venv

**Python Dependencies (from requirements.txt):**
- redis >= 6.0.0
- onnxruntime-gpu >= 1.19.0
- mujoco >= 3.0.0
- rich, wandb, termcolor, coloredlogs
- pydelatin, pyfqmr
- tqdm, numpy==1.23.5

**Pre-configured:**
- Isaac Gym installed and configured
- All TWIST2 modules installed (legged_gym, rsl_rl, pose)
- Redis auto-start on container launch
- Pretrained checkpoint ready to use
- Example motions included

### ✅ Verified Capabilities

| Capability | Status | Test Command |
|------------|--------|--------------|
| Environment Setup | ✅ | `bash verify_docker_setup.sh` |
| Simulation (Sim2Sim) | ✅ | `bash sim2sim.sh` |
| Motion Streaming | ✅ | `bash run_motion_server.sh` |
| Training | ✅ | `bash train.sh <exp> cuda:0` |
| Evaluation | ✅ | `bash eval.sh <exp> cuda:0` |
| GUI Interface | ✅ | `bash gui.sh` |
| Real Robot | ⚠️ | `bash sim2real.sh` (requires hardware) |

### 🚀 Quick Start for End Users

```bash
git clone <repo>
cd twist2_docker
chmod +x scripts/*.sh
./scripts/install.sh
./scripts/run.sh
docker exec -it twist2 bash
cd /workspace/twist2
bash sim2sim.sh  # Works immediately!
```

### 📊 Build Stats

- **Build Time:** ~13 minutes (first time)
- **Image Size:** ~8-10 GB
- **Container Size:** ~10-12 GB with workspace
- **Dependencies:** All pre-installed, zero manual setup needed

### 🔧 For Developers

**Updating the Environment:**

```bash
# After modifying Dockerfile or requirements.txt
./scripts/rebuild_docker.sh

# Or manually
docker compose build --no-cache
docker compose up -d --force-recreate
```

**Adding New Dependencies:**

1. Add to `requirements.txt` for Python packages
2. Add to Dockerfile for system packages
3. Run `./scripts/rebuild_docker.sh`

**Testing Changes:**

```bash
docker exec -it twist2 bash
cd /workspace/twist2
bash verify_docker_setup.sh  # Should all pass
bash sim2sim.sh              # Should work
```

### 📝 Documentation Structure

```
twist2_docker/
├── README.md                     # Main documentation
├── QUICK_START.md                # Fast setup guide
├── CHANGELOG.md                  # This file
├── DOCKER_VERIFICATION_STEPS.md  # Testing guide
├── DOCKER_SETUP_NOTES.md         # Technical notes
├── requirements.txt              # Python dependencies
├── scripts/
│   ├── README.md                 # Scripts documentation
│   ├── install.sh                # Installation script
│   ├── run.sh                    # Run script
│   └── rebuild_docker.sh         # Rebuild script
└── twist2/
    ├── verify_docker_setup.sh    # Verification script
    └── VERIFICATION_GUIDE.md     # Capability testing
```

### 🙏 Acknowledgments

- Original TWIST2: https://github.com/amazon-far/TWIST2
- Isaac Gym by NVIDIA
- Community testing and feedback

---

## How to Use This Release

### For End Users:
1. Clone the repository
2. Run `./scripts/install.sh`
3. Run `./scripts/run.sh`
4. Start working immediately!

### For Contributors:
1. All dependencies are in `requirements.txt`
2. Dockerfile is clean and maintainable
3. Helper scripts make testing easy
4. Full verification suite included

---

**Status: Production Ready ✅**

All features tested and verified on Ubuntu 20.04 with NVIDIA RTX 4090.

