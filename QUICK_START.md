# TWIST2 Docker - Quick Start Guide

## For Users (Downloading Your Package)

### One-Time Setup (Using Scripts - Easiest!)
```bash
# 1. Clone the repository
git clone <your-repo-url>
cd twist2_docker

# 2. Make scripts executable
chmod +x scripts/*.sh

# 3. Run installer (handles everything!)
./scripts/install.sh

# 4. Start container
./scripts/run.sh

# 5. Enter and verify
docker exec -it twist2 bash
cd /workspace/twist2
bash verify_docker_setup.sh
```

### Alternative: Manual Setup
```bash
# 1. Clone the repository
git clone <your-repo-url>
cd twist2_docker

# 2. Allow Docker to access display (add to ~/.bashrc for permanent)
echo 'xhost +local:docker >/dev/null 2>&1' >> ~/.bashrc
xhost +local:docker

# 3. Build and start (first time: ~13 minutes)
docker compose build
docker compose up -d

# 4. Enter container
docker exec -it twist2 bash

# 5. Verify everything works
cd /workspace/twist2
bash verify_docker_setup.sh
```

### Test Simulation (2 Terminals)

**Terminal 1:**
```bash
docker exec -it twist2 bash
cd /workspace/twist2
bash sim2sim.sh
```

**Terminal 2:**
```bash
docker exec -it twist2 bash
cd /workspace/twist2
bash run_motion_server.sh
```

**Expected:** Isaac Gym window with robot walking ✅

### Daily Usage
```bash
docker exec -it twist2 bash
cd /workspace/twist2
bash gui.sh  # Or run any other commands
```

---

## For You (Package Maintainer)

### To Deploy the Updated Dockerfile

**Current Status:**
- ✅ All dependencies identified
- ✅ New Dockerfile created (`Dockerfile.new`)
- ✅ Documentation written
- ⏳ Needs testing and deployment

**Steps to Deploy:**

```bash
cd /home/robo/CodeSpace/twist2_docker

# Option A: Using the rebuild script (Easiest!)
chmod +x scripts/*.sh
./scripts/rebuild_docker.sh

# Option B: Manual rebuild
docker compose down --remove-orphans
docker compose build --no-cache
docker compose up -d

# Test the new build
docker exec -it twist2 bash
cd /workspace/twist2
bash verify_docker_setup.sh  # All should pass ✅
bash sim2sim.sh  # Should work immediately ✅
```

**What Changed:**
- Added `redis-server` (system package)
- Added `redis`, `onnxruntime-gpu`, `mujoco` (Python packages)
- Auto-installs `legged_gym`, `rsl_rl`, `pose` modules
- Auto-starts Redis on container start
- Includes verification script

**Before vs After:**

| Action | Old Setup | New Setup |
|--------|-----------|-----------|
| User runs `sim2sim.sh` | ❌ Missing mujoco | ✅ Works immediately |
| User runs verification | ❌ Missing packages | ✅ All pass |
| Redis needed | ❌ Manual install | ✅ Auto-starts |
| ONNX inference | ❌ Not installed | ✅ Pre-installed |
| First-time experience | 😞 Errors, manual fixes | 😊 Works out of box |

---

## Testing Your Current Setup

Want to test without rebuilding? Try this:

**In your current container:**
```bash
# You already have everything installed from manual fixes!
# So sim2sim should work now

cd /workspace/twist2
bash sim2sim.sh
```

Then open another terminal:
```bash
docker exec -it twist2 bash
cd /workspace/twist2
bash run_motion_server.sh
```

If this works ✅, your setup is good and the new Dockerfile will replicate it!

---

## Distribution Checklist

Before sharing your package:

- [ ] Test new Dockerfile on your machine
- [ ] Test on a clean system (if possible)
- [ ] Update main README
- [ ] Add usage examples/videos
- [ ] Tag release version
- [ ] Share on GitHub

---

## Support Commands

```bash
# Check container status
docker ps -a | grep twist2

# View container logs
docker logs twist2

# Container resource usage
docker stats twist2

# Clean everything and start fresh
docker compose down
docker system prune -f
docker compose build --no-cache
docker compose up -d

# Check what's running in container
docker exec -it twist2 ps aux
```

---

## File Structure for Distribution

```
twist2_docker/
├── README.md                      # Main docs (created ✅)
├── QUICK_START.md                 # This file (created ✅)
├── DOCKER_VERIFICATION_STEPS.md   # Detailed steps (created ✅)
├── DOCKER_SETUP_NOTES.md          # Technical notes (created ✅)
├── Dockerfile                     # Use Dockerfile.new
├── Dockerfile.new                 # Ready to use ✅
├── docker-compose.yml             # No changes needed ✅
├── isaacgym/                      # Isaac Gym installation
└── twist2/                        # TWIST2 code
    ├── verify_docker_setup.sh     # Verification (created ✅)
    ├── VERIFICATION_GUIDE.md      # Testing guide (created ✅)
    ├── sim2sim.sh                 # Existing
    ├── train.sh                   # Existing
    └── ...
```

**All documentation is ready!** Just need to:
1. Replace Dockerfile
2. Test
3. Ship! 🚀

---

## Quick Commands Reference

| Task | Command |
|------|---------|
| Start container | `docker compose up -d` |
| Enter container | `docker exec -it twist2 bash` |
| Stop container | `docker compose down` |
| Rebuild | `docker compose build --no-cache` |
| View logs | `docker logs twist2` |
| Multiple terminals | Run `docker exec -it twist2 bash` in each |
| Verify setup | `bash verify_docker_setup.sh` (inside container) |
| Test simulation | `bash sim2sim.sh` (inside container) |
| Run training | `bash train.sh <name> cuda:0` (inside container) |
| Open GUI | `bash gui.sh` (inside container) |

---

**Ready to ship? All the hard work is done!** 🎉

