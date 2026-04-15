# Installation Guide

This guide documents the complete environment setup for StarVLA, verified on a single NVIDIA RTX 4090 (49GB VRAM) with CUDA 12.4.

---

## Prerequisites

- Python 3.10
- NVIDIA GPU with CUDA 12.x driver
- Linux (Ubuntu 20.04+)

---

## Step 1: Install Miniconda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p ~/miniconda3

# Accept Terms of Service
~/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
~/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

---

## Step 2: Create Conda Environment

```bash
~/miniconda3/bin/conda create -n starvla python=3.10 -y
```

---

## Step 3: Install System Dependencies

```bash
# C/C++ compiler (required for some pip packages)
sudo apt-get install -y gcc g++ build-essential
```

---

## Step 4: Clone StarVLA

```bash
git clone https://github.com/starVLA/starVLA.git
cd starVLA
```

---

## Step 5: Install Python Dependencies

```bash
# Install main dependencies
~/miniconda3/envs/starvla/bin/pip install -r requirements.txt

# Install flash-attn (pre-built wheel for torch 2.6 + CUDA 12.4, no nvcc required)
wget "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl" \
  -O "/tmp/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
~/miniconda3/envs/starvla/bin/pip install "/tmp/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

# Install StarVLA package (editable mode)
~/miniconda3/envs/starvla/bin/pip install -e .
```

> **Note:** The pre-built flash-attn wheel above targets `torch==2.6.0 + CUDA 12.x + Python 3.10`.
> If your environment differs, check [flash-attention releases](https://github.com/Dao-AILab/flash-attention/releases) for the matching wheel,
> or build from source: `pip install flash-attn --no-build-isolation` (requires nvcc).

---

## Step 6: Install CUDA nvcc (required by DeepSpeed)

DeepSpeed needs `nvcc` to compile CUDA ops. Install via conda:

```bash
~/miniconda3/bin/conda install -n starvla -c nvidia/label/cuda-12.4.0 cuda-nvcc -y
```

After installation, set `CUDA_HOME` when running training:

```bash
export CUDA_HOME=/home/$USER/miniconda3/envs/starvla
```

---

## Step 7: Verify Installation

```bash
cd starVLA
~/miniconda3/envs/starvla/bin/python -c "
import torch, flash_attn, starVLA
print('torch:', torch.__version__, '| cuda:', torch.cuda.is_available())
print('flash_attn:', flash_attn.__version__)
print('starVLA: OK')
"
```

Expected output:
```
torch: 2.6.0+cu124 | cuda: True
flash_attn: 2.7.4.post1
starVLA: OK
```

---

## Step 8: Download Pretrained Model

We use [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) as the VLM backbone.

```bash
cd starVLA
mkdir -p playground/Pretrained_models

# Use hf-mirror if HuggingFace is not accessible
HF_ENDPOINT=https://hf-mirror.com \
  ~/miniconda3/envs/starvla/bin/huggingface-cli download Qwen/Qwen3-VL-4B-Instruct \
  --local-dir playground/Pretrained_models/Qwen3-VL-4B-Instruct
```

> **Tip:** If the large `.safetensors` files time out, download them directly with wget:
> ```bash
> MODEL_DIR=playground/Pretrained_models/Qwen3-VL-4B-Instruct
> wget -c "https://hf-mirror.com/Qwen/Qwen3-VL-4B-Instruct/resolve/main/model-00001-of-00002.safetensors" -O "$MODEL_DIR/model-00001-of-00002.safetensors"
> wget -c "https://hf-mirror.com/Qwen/Qwen3-VL-4B-Instruct/resolve/main/model-00002-of-00002.safetensors" -O "$MODEL_DIR/model-00002-of-00002.safetensors"
> ```

---

## Step 9: Download LIBERO Dataset

```bash
mkdir -p playground/Datasets/LEROBOT_LIBERO_DATA

# Download libero_goal subset (recommended for single-GPU quick start)
HF_ENDPOINT=https://hf-mirror.com \
  ~/miniconda3/envs/starvla/bin/huggingface-cli download \
  IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot \
  --repo-type dataset \
  --local-dir playground/Datasets/LEROBOT_LIBERO_DATA/libero_goal_no_noops_1.0.0_lerobot

# Copy modality config
cp examples/LIBERO/train_files/modality.json \
  playground/Datasets/LEROBOT_LIBERO_DATA/libero_goal_no_noops_1.0.0_lerobot/meta/
```

> For all 4 LIBERO suites (`libero_all`), also download:
> - `IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot`
> - `IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot`
> - `IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot`

---

## Step 10: Install LIBERO Evaluation Environment (Optional)

Create a separate conda environment for the LIBERO simulator:

```bash
~/miniconda3/bin/conda create -n libero python=3.10 -y
```

Then run the provided install script. **Note:** The script has the StarVLA path hardcoded to `/home/yilunchen/rep/starVLA` and LIBERO will be cloned to `tmp/LIBERO` inside the repo. Edit `STARVLA_DIR` in the script if your path differs.

```bash
bash examples/LIBERO/eval_files/install_libero_local.sh
```

The script will:
1. Install `mujoco` into the `libero` conda environment
2. Clone [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) to `tmp/LIBERO/` and install it
3. Install eval dependencies: `tyro`, `matplotlib`, `mediapy`, `websockets`, `msgpack`, `numpy==1.24.4`
4. Verify the installation by importing `libero`, `mujoco`, `tyro`, and `websockets`

---

## Known Issues & Fixes

### `CUDA_HOME does not exist` (DeepSpeed)
DeepSpeed requires `nvcc`. Fix: install `cuda-nvcc` via conda (Step 6) and set `CUDA_HOME`.

### `flash-attn` build fails (no nvcc)
Use the pre-built wheel from Step 5 instead of building from source.

### HuggingFace download timeout on large files
Use `wget -c` (with resume support) to download `.safetensors` files directly from `hf-mirror.com`.

### `wrist_image` not found in dataset during training
The LIBERO LeRobot dataset includes both `observation.images.image` (primary) and `observation.images.wrist_image` (wrist camera).
The `modality.json` must include both `primary_image` and `wrist_image` entries.
If you see this error, ensure the `modality.json` in each dataset's `meta/` directory matches `examples/LIBERO/train_files/modality.json`.

### EGL cleanup warning during LIBERO evaluation
```
EGLError: EGL_NOT_INITIALIZED, baseOperation = eglMakeCurrent
```
This is a harmless warning that occurs when MuJoCo environments are destroyed at the end of each episode.
It has been suppressed in `tmp/robosuite/robosuite/renderers/context/egl_context.py` and `binding_utils.py`.
Always set `MUJOCO_GL=egl` and `PYOPENGL_PLATFORM=egl` when running headless evaluation.
