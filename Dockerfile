FROM nvidia/opengl:1.2-glvnd-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# System packages installation
RUN apt-get update && apt-get install -y \
    git \
    wget \
    vim \
    nano \
    python3-dev \
    python3-pip \
    python3-venv \
    python3-tk \
    libgl1 \
    libglfw3 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    mesa-utils \
    libvulkan1 \
    vulkan-utils \
    redis-server \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libgomp1 \
    libglib2.0-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Python virtual env for twist2
RUN python3 -m venv /opt/twist2 && \
    /opt/twist2/bin/pip install --upgrade pip

ENV PATH="/opt/twist2/bin:${PATH}"
ENV ISAACGYM_ROOT="/opt/isaacgym"
ENV PYTHONPATH="/opt/isaacgym/python"

# WandB configuration (disabled by default to avoid login prompts)
ENV WANDB_MODE=disabled
ENV WANDB_DISABLED=true

# IsaacGym installation
COPY isaacgym /opt/isaacgym
WORKDIR /opt/isaacgym/python
RUN pip install -e .

# TWIST2 setup
WORKDIR /workspace
COPY twist2 ./twist2

# Install main dependencies from requirements.txt
COPY requirements.txt /workspace/
RUN pip install -r requirements.txt

# Install TWIST2 requirements if they exist
WORKDIR /workspace/twist2
RUN pip install -r requirements.txt || true

# Install TWIST2 submodules in dependency order
# First rsl_rl (no dependencies on other modules)
RUN cd rsl_rl && pip install -e .

# Then legged_gym (depends on rsl_rl)
RUN cd legged_gym && pip install -e .

# Finally pose
RUN cd pose && pip install -e .

# Create example_motions.yaml for testing without full dataset
RUN echo 'root_path: /workspace/twist2/assets/example_motions\n\
motions:\n\
- file: 0807_yanjie_walk_001.pkl\n\
  weight: 1.0\n\
  description: walking motion\n\
- file: 0807_yanjie_walk_002.pkl\n\
  weight: 1.0\n\
  description: walking motion\n\
- file: 0807_yanjie_walk_003.pkl\n\
  weight: 1.0\n\
  description: walking motion\n\
- file: 0807_yanjie_walk_004.pkl\n\
  weight: 1.0\n\
  description: walking motion\n\
- file: 0807_yanjie_walk_005.pkl\n\
  weight: 1.0\n\
  description: walking motion\n\
- file: 0807_yanjie_walk_006.pkl\n\
  weight: 1.0\n\
  description: walking motion\n\
- file: 0807_yanjie_walk_007.pkl\n\
  weight: 1.0\n\
  description: walking motion\n\
- file: 0807_yanjie_walk_008.pkl\n\
  weight: 1.0\n\
  description: walking motion\n\
- file: 0807_yanjie_walk_009.pkl\n\
  weight: 1.0\n\
  description: walking motion\n\
- file: 0807_yanjie_walk_010.pkl\n\
  weight: 1.0\n\
  description: walking motion' > legged_gym/motion_data_configs/example_motions.yaml

# Update configs to use example_motions by default
RUN sed -i 's|twist2_dataset\.yaml|example_motions.yaml|g' legged_gym/legged_gym/envs/g1/g1_mimic_future_config.py && \
    sed -i 's|g1_omomo+mocap_static+amass_walk\.yaml|example_motions.yaml|g' legged_gym/legged_gym/envs/g1/g1_mimic_distill_config.py

# Create entrypoint script to start Redis and set WandB env vars
RUN echo '#!/bin/bash\n\
# Start Redis\n\
redis-server --daemonize yes\n\
\n\
# Set WandB to disabled mode\n\
export WANDB_MODE=disabled\n\
export WANDB_DISABLED=true\n\
\n\
exec "$@"' > /entrypoint.sh && \
    chmod +x /entrypoint.sh

WORKDIR /workspace/twist2

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]
