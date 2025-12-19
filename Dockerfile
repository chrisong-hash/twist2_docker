# syntax=docker/dockerfile:1.4
FROM nvidia/opengl:1.2-glvnd-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# System packages installation
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
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
    libgomp1 \
    libglib2.0-dev \
    build-essential \
    cmake \
    pybind11-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install Miniconda for managing Python environments
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda3 && \
    rm Miniconda3-latest-Linux-x86_64.sh

ENV PATH="/opt/miniconda3/bin:${PATH}"
SHELL ["/bin/bash", "-c"]

# Configure conda to auto-accept licenses and create environments
RUN /opt/miniconda3/bin/conda init bash && \
    /opt/miniconda3/bin/conda config --set always_yes yes && \
    /opt/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    /opt/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    /opt/miniconda3/bin/conda create -n twist2 python=3.8 -y && \
    /opt/miniconda3/bin/conda create -n gmr python=3.10 -y

# Activate twist2 environment by default
ENV PATH="/opt/miniconda3/envs/twist2/bin:${PATH}"
ENV ISAACGYM_ROOT="/opt/isaacgym"
ENV PYTHONPATH="/opt/isaacgym/python"

# WandB configuration
ENV WANDB_MODE=disabled
ENV WANDB_DISABLED=true

# IsaacGym installation (using twist2 env)
COPY isaacgym /opt/isaacgym
WORKDIR /opt/isaacgym/python
RUN source /opt/miniconda3/etc/profile.d/conda.sh && \
    conda activate twist2 && \
    pip install -e .

# TWIST2 setup and external dependencies
WORKDIR /workspace
COPY twist2 ./twist2
COPY unitree_sdk2 ./unitree_sdk2
COPY GMR ./GMR

# Install main dependencies from requirements.txt
COPY requirements.txt /workspace/
RUN source /opt/miniconda3/etc/profile.d/conda.sh && \
    conda activate twist2 && \
    pip install -r requirements.txt

# Install TWIST2 requirements
WORKDIR /workspace/twist2
RUN source /opt/miniconda3/etc/profile.d/conda.sh && \
    conda activate twist2 && \
    pip install -r requirements.txt || true

# Install TWIST2 submodules in dependency order
RUN source /opt/miniconda3/etc/profile.d/conda.sh && \
    conda activate twist2 && \
    cd rsl_rl && pip install -e .
RUN source /opt/miniconda3/etc/profile.d/conda.sh && \
    conda activate twist2 && \
    cd legged_gym && pip install -e .
RUN source /opt/miniconda3/etc/profile.d/conda.sh && \
    conda activate twist2 && \
    cd pose && pip install -e .

# Create example_motions.yaml
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

# Build Unitree SDK with Python binding (using twist2 env)
WORKDIR /workspace/unitree_sdk2
RUN source /opt/miniconda3/etc/profile.d/conda.sh && \
    conda activate twist2 && \
    mkdir -p build && cd build && \
    cmake .. -DBUILD_PYTHON_BINDING=ON && \
    make -j$(nproc) && \
    SITE_PACKAGES=$(/opt/miniconda3/envs/twist2/bin/python -c "import site; print(site.getsitepackages()[0])") && \
    cp lib/unitree_interface.cpython-*-linux-gnu.so $SITE_PACKAGES/unitree_interface.so

# Install GMR (using gmr env)
WORKDIR /workspace/GMR
RUN source /opt/miniconda3/etc/profile.d/conda.sh && \
    conda activate gmr && \
    pip install -e . && \
    conda install -c conda-forge libstdcxx-ng -y && \
    pip install pyyaml

# Install PyTorch for LocoMode (separate layer for caching)
RUN --mount=type=cache,target=/root/.cache/pip \
    source /opt/miniconda3/etc/profile.d/conda.sh && \
    conda activate gmr && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install PICO SDK (XRoboToolkit) in gmr env
WORKDIR /workspace
RUN git clone https://github.com/XR-Robotics/XRoboToolkit-PC-Service-Pybind.git && \
    cd XRoboToolkit-PC-Service-Pybind && \
    mkdir -p tmp && cd tmp && \
    git clone https://github.com/XR-Robotics/XRoboToolkit-PC-Service.git && \
    cd XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK && \
    bash build.sh && \
    cd ../../../.. && \
    mkdir -p lib include && \
    cp tmp/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/PXREARobotSDK.h include/ && \
    cp -r tmp/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/nlohmann include/ && \
    cp tmp/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/build/libPXREARobotSDK.so lib/ && \
    rm -rf tmp && \
    source /opt/miniconda3/etc/profile.d/conda.sh && \
    conda activate gmr && \
    conda install -c conda-forge pybind11 -y && \
    pip install -e .

# Auto-activate twist2 environment in bashrc
RUN echo 'source /opt/miniconda3/etc/profile.d/conda.sh' >> /root/.bashrc && \
    echo 'conda activate twist2' >> /root/.bashrc && \
    echo 'export WANDB_MODE=disabled' >> /root/.bashrc && \
    echo 'export WANDB_DISABLED=true' >> /root/.bashrc

# Create entrypoint script
RUN echo '#!/bin/bash\n\
redis-server --daemonize yes\n\
export WANDB_MODE=disabled\n\
export WANDB_DISABLED=true\n\
source /opt/miniconda3/etc/profile.d/conda.sh\n\
conda activate twist2\n\
exec "$@"' > /entrypoint.sh && \
    chmod +x /entrypoint.sh

WORKDIR /workspace/twist2

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]
