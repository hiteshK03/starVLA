# Getting Started: Single-GPU Training & Inference

This guide covers single-GPU training and inference for StarVLA on the LIBERO benchmark.
For full installation, see [installation.md](installation.md).

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Single-GPU Training](#single-gpu-training)
- [Single-GPU Inference](#single-gpu-inference)

---

## Prerequisites

Before running training or inference, ensure:

1. ✅ Environment installed — see [installation.md](installation.md)
2. ✅ Pretrained model downloaded to `playground/Pretrained_models/Qwen3-VL-4B-Instruct/`
3. ✅ LIBERO dataset downloaded to `playground/Datasets/LEROBOT_LIBERO_DATA/`
4. ✅ `modality.json` copied to the dataset's `meta/` directory

Verify your setup:

```bash
cd starVLA

# Check model files
ls playground/Pretrained_models/Qwen3-VL-4B-Instruct/*.safetensors

# Check dataset
ls playground/Datasets/LEROBOT_LIBERO_DATA/libero_goal_no_noops_1.0.0_lerobot/meta/

# Verify dataloader
~/miniconda3/envs/starvla/bin/python starVLA/dataloader/lerobot_datasets.py \
  --config_yaml examples/LIBERO/train_files/starvla_cotrain_libero.yaml
```

---

## Single-GPU Training

### Quick Start

A ready-to-use single-GPU training script is provided at:
`examples/LIBERO/train_files/bar/run_libero_train_single_gpu.sh`

```bash
cd starVLA
nohup bash examples/LIBERO/train_files/bar/run_libero_train_single_gpu.sh \
  > /tmp/train_full.log 2>&1 &

echo "Training PID: $!"
tail -f /tmp/train_full.log
```

### Key Adaptations for Single GPU (RTX 4090 / 49GB VRAM)

| Parameter | Multi-GPU Default | Single-GPU Setting | Reason |
|-----------|------------------|--------------------|--------|
| `--num_processes` | 8 | **1** | Single GPU |
| `per_device_batch_size` | 16 | **1** | VRAM constraint |
| `freeze_modules` | `''` | **`qwen_vl_interface`** | Freeze VLM backbone to save VRAM |
| `PYTORCH_CUDA_ALLOC_CONF` | — | **`expandable_segments:True`** | Avoid memory fragmentation |
| `CUDA_HOME` | system | **`~/miniconda3/envs/starvla`** | DeepSpeed nvcc requirement |

### Manual Launch

```bash
cd starVLA
export CUDA_HOME=~/miniconda3/envs/starvla
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

~/miniconda3/envs/starvla/bin/accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 1 \
  starVLA/training/train_starvla.py \
  --config_yaml examples/LIBERO/train_files/starvla_cotrain_libero.yaml \
  --framework.name QwenOFT \
  --framework.qwenvl.base_vlm playground/Pretrained_models/Qwen3-VL-4B-Instruct \
  --datasets.vla_data.data_root_dir playground/Datasets/LEROBOT_LIBERO_DATA \
  --datasets.vla_data.data_mix libero_goal \
  --datasets.vla_data.per_device_batch_size 1 \
  --datasets.vlm_data.per_device_batch_size 1 \
  --trainer.freeze_modules qwen_vl_interface \
  --trainer.max_train_steps 10000 \
  --trainer.save_interval 2000 \
  --trainer.logging_frequency 50 \
  --trainer.eval_interval 100 \
  --run_root_dir results/Checkpoints \
  --run_id single_gpu_libero_goal_qwen3oft \
  --wandb_project starVLA_Libero
```

### Debug Mode (5 steps smoke test)

Before full training, run a quick 5-step debug to verify everything works:

```bash
cd starVLA
export CUDA_HOME=~/miniconda3/envs/starvla
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

~/miniconda3/envs/starvla/bin/accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 1 \
  starVLA/training/train_starvla.py \
  --config_yaml examples/LIBERO/train_files/starvla_cotrain_libero.yaml \
  --framework.name QwenOFT \
  --framework.qwenvl.base_vlm playground/Pretrained_models/Qwen3-VL-4B-Instruct \
  --datasets.vla_data.data_root_dir playground/Datasets/LEROBOT_LIBERO_DATA \
  --datasets.vla_data.data_mix libero_goal \
  --datasets.vla_data.per_device_batch_size 1 \
  --datasets.vlm_data.per_device_batch_size 1 \
  --trainer.freeze_modules qwen_vl_interface \
  --trainer.max_train_steps 5 \
  --trainer.save_interval 5 \
  --trainer.logging_frequency 1 \
  --run_root_dir results/Checkpoints \
  --run_id debug_test \
  --is_debug True
```

Expected output:
```
Step 1, Loss: {'action_dit_loss': ...}
Step 2, Loss: {'action_dit_loss': ...}
...
Step 5, Loss: {'action_dit_loss': ...}
✅ Checkpoint saved at results/Checkpoints/debug_test/checkpoints/steps_5
```

### Training Output

Checkpoints are saved to:
```
results/Checkpoints/single_gpu_libero_goal_qwen3oft/
└── checkpoints/
    ├── steps_2000_pytorch_model.pt
    ├── steps_4000_pytorch_model.pt
    ├── steps_6000_pytorch_model.pt
    ├── steps_8000_pytorch_model.pt
    └── steps_10000_pytorch_model.pt
```

### Switching VLA Framework

Change `--framework.name` to use a different architecture:

| Framework | `--framework.name` | Description |
|-----------|-------------------|-------------|
| StarVLA-OFT | `QwenOFT` | MLP action head (fastest, recommended for single GPU) |
| StarVLA-GR00T | `QwenGR00T` | Dual-system: VLM + Flow-matching |
| StarVLA-π | `QwenPI` | Flow-matching diffusion action head |
| StarVLA-FAST | `QwenFAST` | Discrete autoregressive action tokens |

---

## Single-GPU Inference

Inference uses a **client-server architecture**:
- **Policy Server** (starvla env): loads the checkpoint and serves action predictions via WebSocket
- **Eval Client** (libero env): runs the LIBERO simulation and queries the server for actions

### Step 0: Download Pretrained Checkpoint

```bash
cd starVLA
HF_ENDPOINT=https://hf-mirror.com ~/miniconda3/envs/starvla/bin/python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='StarVLA/Qwen3-VL-OFT-LIBERO-4in1',
    filename='checkpoints/steps_50000_pytorch_model.pt',
    local_dir='playground/Pretrained_models/StarVLA/Qwen3-VL-OFT-LIBERO-4in1',
    resume_download=True,
)
print('Download complete!')
"
```

The checkpoint will be saved to:
```
playground/Pretrained_models/StarVLA/Qwen3-VL-OFT-LIBERO-4in1/
├── checkpoints/
│   └── steps_50000_pytorch_model.pt   # ~9.2 GB
├── config.yaml
└── dataset_statistics.json
```

### Step 1: Start Policy Server

Open **Terminal 1** and run:

```bash
cd starVLA
export PYTHONPATH=$(pwd):${PYTHONPATH}
export CUDA_HOME=~/miniconda3/envs/starvla

CUDA_VISIBLE_DEVICES=0 ~/miniconda3/envs/starvla/bin/python \
    deployment/model_server/server_policy.py \
    --ckpt_path playground/Pretrained_models/StarVLA/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt \
    --port 6694 \
    --use_bf16
```

Wait until you see:
```
INFO:root:server running ...
INFO:websockets.server:server listening on 0.0.0.0:6694
```

Or use the provided script:
```bash
bash examples/LIBERO/eval_files/run_policy_server_local.sh
```

### Step 2: Run LIBERO Evaluation

Open **Terminal 2** and run:

```bash
cd starVLA
export LIBERO_HOME=/home/$USER/rep/starVLA/tmp/LIBERO
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
export PYTHONPATH=${LIBERO_HOME}:$(pwd):${PYTHONPATH}
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EGL_DEVICE_ID=0   # specify GPU device for headless EGL rendering

~/miniconda3/envs/libero/bin/python ./examples/LIBERO/eval_files/eval_libero.py \
    --args.pretrained-path playground/Pretrained_models/StarVLA/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt \
    --args.host "127.0.0.1" \
    --args.port 6694 \
    --args.task-suite-name "libero_goal" \
    --args.num-trials-per-task 50 \
    --args.video-out-path "results/libero_goal/steps_50000"
```

Or use the provided script:
```bash
bash examples/LIBERO/eval_files/eval_libero_local.sh
```

> **Important:** The policy server must remain running in Terminal 1 throughout the entire evaluation. Do not close Terminal 1.

### Evaluation Parameters

| Parameter | Description | Options |
|-----------|-------------|---------|
| `--args.task-suite-name` | Task suite to evaluate | `libero_spatial`, `libero_object`, `libero_goal`, `libero_10` |
| `--args.num-trials-per-task` | Rollouts per task | `5` (quick test), `50` (full eval) |
| `--args.video-out-path` | Directory to save rollout videos | any path |

### Expected Results (Qwen3-VL-OFT-LIBERO-4in1, 50k steps)

| Task Suite | Success Rate |
|------------|-------------|
| libero_object | 99.6% |
| libero_spatial | 99.0% |
| libero_goal | 98.6% |
| libero_10 | 94.8% |
| **Average** | **98.0%** |

### Output

Rollout videos are saved to `results/libero_goal/steps_50000/`:
```
results/libero_goal/steps_50000/
├── rollout_put_the_bowl_on_the_plate_episode0_success.mp4
├── rollout_put_the_bowl_on_the_plate_episode1_success.mp4
└── ...
```


