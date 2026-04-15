# Getting Started: RoboTwin 2.0 Training & Evaluation

This guide covers single-GPU training and evaluation for StarVLA on the RoboTwin 2.0 benchmark.
For full installation, see [installation_robotwin.md](installation_robotwin.md).

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Single-GPU Training](#single-gpu-training)
- [Single-GPU Evaluation](#single-gpu-evaluation)
- [Downloading Pretrained Checkpoints](#downloading-pretrained-checkpoints)

---

## Prerequisites

Before running training or evaluation, ensure:

1. ✅ StarVLA environment installed — see [installation.md](installation.md)
2. ✅ RoboTwin environment installed — see [installation_robotwin.md](installation_robotwin.md)
3. ✅ Pretrained model downloaded to `playground/Pretrained_models/Qwen3-VL-4B-Instruct/`
4. ✅ RoboTwin 2.0 Clean dataset downloaded to `playground/Datasets/RoboTwin/Clean/`
5. ✅ `modality.json` copied to each task's `meta/` directory
6. ✅ `ROBOTWIN_PATH` set to your local RoboTwin checkout (for evaluation)

Verify your setup:

```bash
cd starVLA

# Check model files
ls playground/Pretrained_models/Qwen3-VL-4B-Instruct/*.safetensors

# Check dataset (should list 50 task directories)
ls playground/Datasets/RoboTwin/Clean/ | wc -l

# Check LeRobot format conversion (meta/info.json must exist)
ls playground/Datasets/RoboTwin/Clean/adjust_bottle/aloha-agilex_clean_50/meta/info.json
ls playground/Datasets/RoboTwin/Clean/adjust_bottle/aloha-agilex_clean_50/data/chunk-000/ | head -3
```

---

## Single-GPU Training

### Quick Start: Debug Smoke Test (5 steps)

Before full training, run a 5-step debug to verify data loading and forward pass work correctly:

```bash
cd starVLA
export CUDA_HOME=~/miniconda3/envs/starvla
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

IS_DEBUG=true bash examples/Robotwin/train_files/run_robotwin_train_single_gpu.sh
```

Expected output:
```
Step 1, Loss: {'action_dit_loss': ...}
Step 2, Loss: {'action_dit_loss': ...}
...
Step 5, Loss: {'action_dit_loss': ...}
Checkpoint saved at results/Checkpoints/debug_robotwin_single_gpu/checkpoints/steps_5
```

### Quick Start: Single-Task Training (1000 steps)

Train on a single task (`adjust_bottle`) to quickly validate the full training pipeline:

```bash
cd starVLA
export CUDA_HOME=~/miniconda3/envs/starvla
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA_MIX=robotwin_task1 \
RUN_ID=single_task_adjust_bottle \
bash examples/Robotwin/train_files/run_robotwin_train_single_gpu.sh
```

### Full Single-GPU Training (Clean dataset, 50 tasks)

Train on all 50 tasks using the Clean dataset (50 × 50 = 2500 trajectories):

```bash
cd starVLA
export CUDA_HOME=~/miniconda3/envs/starvla
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA_MIX=robotwin_clean_50 \
RUN_ID=single_gpu_robotwin_clean \
bash examples/Robotwin/train_files/run_robotwin_train_single_gpu.sh
```

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
  --config_yaml examples/Robotwin/train_files/starvla_cotrain_robotwin_abs.yaml \
  --framework.name QwenOFT \
  --framework.qwenvl.base_vlm playground/Pretrained_models/Qwen3-VL-4B-Instruct \
  --datasets.vla_data.data_root_dir playground/Datasets/RoboTwin \
  --datasets.vla_data.data_mix robotwin_clean_50 \
  --datasets.vla_data.per_device_batch_size 1 \
  --datasets.vlm_data.per_device_batch_size 1 \
  --datasets.vla_data.action_type abs_qpos \
  --datasets.vla_data.action_mode abs \
  --trainer.freeze_modules qwen_vl_interface \
  --trainer.max_train_steps 10000 \
  --trainer.save_interval 2000 \
  --trainer.logging_frequency 50 \
  --trainer.eval_interval 500 \
  --run_root_dir results/Checkpoints \
  --run_id single_gpu_robotwin_clean \
  --wandb_project starVLA_Robotwin
```

### Key Adaptations for Single GPU

| Parameter | Multi-GPU Default | Single-GPU Setting | Reason |
|-----------|------------------|--------------------|--------|
| `--num_processes` | 8 | **1** | Single GPU |
| `per_device_batch_size` (vla) | 4 | **1** | VRAM constraint |
| `per_device_batch_size` (vlm) | 4 | **1** | VRAM constraint |
| `freeze_modules` | `''` | **`qwen_vl_interface`** | Freeze VLM backbone to save VRAM |
| `PYTORCH_CUDA_ALLOC_CONF` | — | **`expandable_segments:True`** | Avoid memory fragmentation |
| `CUDA_HOME` | system | **`~/miniconda3/envs/starvla`** | DeepSpeed nvcc requirement |

### Training Output

Checkpoints are saved to:
```
results/Checkpoints/<run_id>/
└── checkpoints/
    ├── steps_2000_pytorch_model.pt
    ├── steps_4000_pytorch_model.pt
    ├── steps_6000_pytorch_model.pt
    ├── steps_8000_pytorch_model.pt
    └── steps_10000_pytorch_model.pt
```

---

## Single-GPU Evaluation

Evaluation uses a **client-server architecture**:
- **Policy Server** (`starvla` env): loads the checkpoint and serves action predictions via WebSocket
- **Eval Client** (`robotwin` env): runs the RoboTwin simulation and queries the server for actions

### Prerequisites for Evaluation

```bash
# Set RoboTwin path (required)
export ROBOTWIN_PATH=/path/to/RoboTwin

# Verify the eval_policy.py patch has been applied
grep -n "policy_ckpt_path" ${ROBOTWIN_PATH}/script/eval_policy.py | head -3
# Should show 3 matches if patch is applied
```

### Recommended: One-Command Evaluation

Use the provided wrapper script for single-GPU evaluation:

```bash
cd starVLA
export ROBOTWIN_PATH=/path/to/RoboTwin

bash examples/Robotwin/eval_files/run_robotwin_eval_single_gpu.sh \
    --ckpt /path/to/checkpoint.pt \
    --task adjust_bottle \
    --mode demo_clean \
    --name my_eval
```

### Using `start_eval.sh` Directly

```bash
cd starVLA
export ROBOTWIN_PATH=/path/to/RoboTwin

# Single task, clean mode
bash examples/Robotwin/eval_files/start_eval.sh \
    -m demo_clean \
    -n test_eval \
    -c /path/to/checkpoint.pt \
    adjust_bottle
```

Multiple tasks:

```bash
bash examples/Robotwin/eval_files/start_eval.sh \
    -m demo_clean \
    -n my_run \
    -c /path/to/checkpoint.pt \
    adjust_bottle open_laptop lift_pot place_shoe
```

All 50 tasks:

```bash
bash examples/Robotwin/eval_files/start_eval.sh \
    -m demo_clean \
    -n full_eval \
    -c /path/to/checkpoint.pt \
    all
```

### Low-Level Manual Mode

If you prefer to manage the policy server and eval processes yourself:

**Terminal 1** — Start the policy server (`starvla` env):

```bash
cd starVLA
export PYTHONPATH=$(pwd):${PYTHONPATH}
export CUDA_HOME=~/miniconda3/envs/starvla

bash examples/Robotwin/eval_files/run_policy_server.sh \
    /path/to/checkpoint.pt \
    0 \
    5694
```

Wait until you see:
```
INFO:root:server running ...
INFO:websockets.server:server listening on 0.0.0.0:5694
```

**Terminal 2** — Run evaluation (`robotwin` env):

```bash
cd starVLA/examples/Robotwin/eval_files
bash eval.sh adjust_bottle demo_clean my_eval 0 0 /path/to/checkpoint.pt 5694
```

### Evaluation Output

Real-time per-episode success rates are streamed to stdout:

```
[RESULT] adjust_bottle: Success rate: 1/1 => 100.0%, current seed: 100001
[RESULT] adjust_bottle: Success rate: 2/2 => 100.0%, current seed: 100002
[RESULT] adjust_bottle: Success rate: 3/3 => 100.0%, current seed: 100005
```

Full logs are saved to:
```
<ckpt_dir>/robotwin_eval_logs/<name>_<mode>_<ckpt_stem>_<timestamp>/
    <task>_<mode>_slot0_gpu0_port5694_server.log
    <task>_<mode>_slot0_gpu0_port5694_eval.log
```

---

## Downloading Pretrained Checkpoints

### Clean-only checkpoint (50 tasks × 50 demos)

```bash
cd starVLA
HF_ENDPOINT=https://hf-mirror.com \
  ~/miniconda3/envs/starvla/bin/python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='StarVLA/Qwen3-VL-OFT-Robotwin2',
    filename='checkpoints/steps_50000_pytorch_model.pt',
    local_dir='playground/Pretrained_models/StarVLA/Qwen3-VL-OFT-Robotwin2',
    resume_download=True,
)
print('Download complete!')
"
```

### Full checkpoint (50 tasks × 550 demos, clean + randomized)

```bash
cd starVLA
HF_ENDPOINT=https://hf-mirror.com \
  ~/miniconda3/envs/starvla/bin/python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='StarVLA/Qwen3-VL-OFT-RoboTwin2-All',
    filename='checkpoints/steps_150000_pytorch_model.pt',
    local_dir='playground/Pretrained_models/StarVLA/Qwen3-VL-OFT-RoboTwin2-All',
    resume_download=True,
)
print('Download complete!')
"
```

### Expected Results (Pretrained Checkpoints)

**Clean-only model** (50 × 50 demos):

| Metric | Value |
|--------|-------|
| Average Easy | 88.18% |
| Average Hard | 88.32% |

**Full model** (50 × 550 demos, clean + randomized):

| Metric | Value |
|--------|-------|
| Average Easy | 88.66% |
| Average Hard | 87.02% |

See [examples/Robotwin/README.md](../examples/Robotwin/README.md) for per-task results.

---

## Troubleshooting

### Training: `CUDA_HOME does not exist`
```bash
export CUDA_HOME=~/miniconda3/envs/starvla
```

### Training: Out of Memory (OOM)
- Ensure `freeze_modules=qwen_vl_interface` is set (freezes VLM backbone)
- Reduce `per_device_batch_size` to 1
- Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

### Evaluation: `ROBOTWIN_PATH` not set
```bash
export ROBOTWIN_PATH=/path/to/RoboTwin
```

### Evaluation: `policy_ckpt_path` KeyError
Apply the `eval_policy.py` patch described in [installation_robotwin.md](installation_robotwin.md#step-6-clone-robotwin-repository-and-apply-patch).

### Evaluation: Policy server timeout
Increase the server timeout:
```bash
bash examples/Robotwin/eval_files/start_eval.sh \
    --server-timeout 1200 \
    -m demo_clean -n test -c /path/to/ckpt.pt \
    adjust_bottle
```
