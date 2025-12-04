# TWIST2 Docker Scripts

Helper scripts for managing the TWIST2 Docker environment.

## Scripts Overview

### 🚀 install.sh
**First-time setup script**

Checks prerequisites, sets up X11 access, and builds the Docker image.

```bash
./scripts/install.sh
```

**What it does:**
- ✅ Checks Docker installation
- ✅ Checks NVIDIA Docker support
- ✅ Sets up X11 access (with option to make persistent)
- ✅ Builds the Docker image

**Run once when setting up the project.**

---

### ▶️ run.sh
**Start or enter the container**

Starts the container if stopped, or shows how to enter if already running.

```bash
./scripts/run.sh
```

**What it does:**
- Creates container if it doesn't exist
- Starts container if it's stopped
- Shows commands to enter the running container

**Run whenever you want to start working.**

---

### 🔄 rebuild_docker.sh
**Rebuild from scratch**

Completely rebuilds the Docker image with a clean slate.

```bash
./scripts/rebuild_docker.sh
```

**What it does:**
- Stops and removes existing container
- Optionally cleans up old images
- Rebuilds Docker image with --no-cache
- Starts fresh container

**Run when:**
- Dockerfile has changed
- Dependencies need updating
- Something is broken and needs fresh start

---

## Quick Workflow

### First Time Setup
```bash
# 1. Install everything
./scripts/install.sh

# 2. Start container
./scripts/run.sh

# 3. Enter and verify
docker exec -it twist2 bash
cd /workspace/twist2
bash verify_docker_setup.sh
```

### Daily Usage
```bash
# Start container (if stopped)
./scripts/run.sh

# Enter container
docker exec -it twist2 bash

# Work in container
cd /workspace/twist2
bash sim2sim.sh
```

### After Updates
```bash
# Rebuild everything
./scripts/rebuild_docker.sh

# Verify
docker exec -it twist2 bash
cd /workspace/twist2
bash verify_docker_setup.sh
```

---

## Making Scripts Executable

If you get "permission denied", run:

```bash
chmod +x scripts/*.sh
```

---

## Manual Commands

If you prefer to run Docker commands directly:

```bash
# Build
docker compose build

# Start
docker compose up -d

# Enter
docker exec -it twist2 bash

# Stop
docker compose down

# Rebuild
docker compose build --no-cache
docker compose up -d --force-recreate
```

---

## Troubleshooting

### "Docker not found"
Install Docker: https://docs.docker.com/engine/install/

### "NVIDIA Docker not working"
Install nvidia-docker2: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

### "Cannot open display"
Run on host: `xhost +local:docker`

### "Container already exists"
```bash
docker compose down
./scripts/run.sh
```

---

**See main README.md for more information.**

