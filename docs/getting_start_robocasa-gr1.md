# 🚀 Getting Started: RoboCasa GR-1 + StarVLA

This guide covers **single-GPU training** and **single-GPU evaluation** for the RoboCasa GR-1 Tabletop Tasks benchmark.

> **Before you begin**: Make sure you have completed [Installation Guide](./installation_robocasa-gr1.md) — both conda environments (`starVLA` and `robocasa`) must be ready and the dataset must be downloaded.

---

## Quick Overview

The workflow involves **two separate conda environments** running in parallel during evaluation:

```
Terminal A (starVLA env)          Terminal B (robocasa env)
─────────────────────────         ─────────────────────────
Policy Server (GPU)          ←→   Simulation Client (CPU/GPU)
deployment/model_server/          examples/Robocasa_tabletop/
  server_policy.py                  eval_files/simulation_env.py
```

During training, only the `starVLA` environment is needed.

---

## Part 1 — Single-GPU Training

### Step 1: Activate the StarVLA environment

```bash
conda activate starVLA
cd /path/to/starVLA   # project root
```

### Step 2: Verify the dataset is in place

```bash
ls playground/Datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/ | head -5
# Expected: gr1_unified.PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_1000  ...
```

### Step 3: Run single-GPU training

A ready-to-use single-GPU script is provided:

```bash
bash examples/Robocasa_tabletop/train_files/run_robocasa_single_gpu.sh
```

Key differences from the 8-GPU script (`run_robocasa.sh`):

| Parameter | 8-GPU script | Single-GPU script |
|-----------|-------------|-------------------|
| `--num_processes` | 8 | **1** |
| `--datasets.vla_data.per_device_batch_size` | 8 | **2** |
| `--datasets.vlm_data.per_device_batch_size` | 4 | **2** |
| NCCL settings | enabled | **disabled** |

### Step 4: Monitor training

Checkpoints are saved every 10,000 steps to:

```
playground/Checkpoints/debug_starvla_single_gpu_robocasa_gr1/
└── checkpoints/
    ├── steps_10000_pytorch_model.pt
    ├── steps_20000_pytorch_model.pt
    └── ...
```

Training logs are printed to stdout. To enable W&B logging, edit the script and set:

```bash
export WANDB_MODE=online
# and fill in --wandb_entity your_entity
```

### Customising the training script

Open `examples/Robocasa_tabletop/train_files/run_robocasa_single_gpu.sh` and adjust:

```bash
# Change the base VLM backbone
base_vlm=./playground/Pretrained_models/Qwen3-VL-4B-Instruct

# Change which task mix to train on
data_mix=fourier_gr1_unified_1000

# Reduce steps for a quick smoke-test
--trainer.max_train_steps 500
--is_debug True
```

---

## Part 2 — Single-GPU Evaluation

Evaluation requires **two terminals** running simultaneously.

### Option A: Automated script (recommended)

A convenience script handles both the server and the simulation in one command:

```bash
conda activate starVLA   # the script internally calls both envs

bash examples/Robocasa_tabletop/eval_files/eval_single_gpu.sh \
    ./playground/Pretrained_models/StarVLA/Qwen3-VL-OFT-Robocasa/checkpoints/steps_90000_pytorch_model.pt \
    "gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env" \
    10 \
    5678
```

Arguments (all optional, defaults shown):

| Position | Argument | Default |
|----------|----------|---------|
| 1 | `CKPT_PATH` | `./playground/Pretrained_models/StarVLA/Qwen3-VL-OFT-Robocasa/checkpoints/steps_90000_pytorch_model.pt` |
| 2 | `ENV_NAME` | `gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env` |
| 3 | `N_EPISODES` | `10` |
| 4 | `PORT` | `5678` |

Videos are saved automatically to:
```
<checkpoint_dir>/videos/<ckpt_name>/single_gpu_<env_name>/
```

---

### Option B: Manual two-terminal workflow

#### Terminal A — Start the policy server (starVLA env)

```bash
conda activate starVLA
cd /path/to/starVLA

CKPT_PATH="./playground/Pretrained_models/StarVLA/Qwen3-VL-OFT-Robocasa/checkpoints/steps_90000_pytorch_model.pt"

python deployment/model_server/server_policy.py \
    --ckpt_path "${CKPT_PATH}" \
    --port 5678 \
    --use_bf16
```

Wait until you see a message like `Server started on port 5678` before proceeding.

#### Terminal B — Run the simulation (robocasa env)

```bash
conda activate robocasa
cd /path/to/starVLA

export PYTHONPATH=$(pwd):${PYTHONPATH}

ENV_NAME="gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env"
CKPT_PATH="./playground/Pretrained_models/StarVLA/Qwen3-VL-OFT-Robocasa/checkpoints/steps_90000_pytorch_model.pt"

python examples/Robocasa_tabletop/eval_files/simulation_env.py \
    --args.env_name "${ENV_NAME}" \
    --args.port 5678 \
    --args.n_episodes 10 \
    --args.n_envs 1 \
    --args.max_episode_steps 720 \
    --args.n_action_steps 12 \
    --args.video_out_path ./videos/test_run \
    --args.pretrained_path "${CKPT_PATH}"
```

---

## Part 3 — Available Environments

The following 24 task environments are supported:

**Closed-container tasks** (6 tasks):
```
gr1_unified/PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env
gr1_unified/PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env
gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env
gr1_unified/PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env
gr1_unified/PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env
gr1_unified/PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_Env
```

**Novel object generalisation tasks** (18 tasks, from Cuttingboard / Placemat / Plate / Tray):
```
gr1_unified/PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_Env
gr1_unified/PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env
gr1_unified/PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_Env
gr1_unified/PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_Env
gr1_unified/PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env
gr1_unified/PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_Env
gr1_unified/PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_Env
gr1_unified/PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_Env
gr1_unified/PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env
gr1_unified/PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env
gr1_unified/PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env
gr1_unified/PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_Env
gr1_unified/PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env
gr1_unified/PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env
gr1_unified/PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_Env
gr1_unified/PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_Env
gr1_unified/PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env
gr1_unified/PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env
```

---

## Part 4 — Multi-GPU Evaluation (Optional)

If you have multiple GPUs, use the batch evaluation script to run all 24 tasks in parallel:

```bash
conda activate starVLA
cd /path/to/starVLA

bash examples/Robocasa_tabletop/eval_files/batch_eval_args.sh \
    ./playground/Pretrained_models/StarVLA/Qwen3-VL-OFT-Robocasa/checkpoints/steps_90000_pytorch_model.pt
```

> ⚠️ Edit `batch_eval_args.sh` to set the correct `starVLA_PYTHON` and `ROBOCASA_PYTHON` paths before running.

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `Connection refused` on port 5678 | Policy server not ready yet | Wait longer (up to 60 s) for model to load |
| `CUDA out of memory` during training | Batch size too large | Reduce `per_device_batch_size` to 1 in the training script |
| `ModuleNotFoundError: examples` | `PYTHONPATH` not set | Run `export PYTHONPATH=$(pwd):${PYTHONPATH}` from the project root |
| Simulation hangs at episode 0 | Server crashed silently | Check Terminal A for error messages |
| Low success rate | Wrong `n_action_steps` | Try `--args.n_action_steps 12` (default for GR-1) |
| `robocasa assets not found` | Assets not downloaded | Run `python -m robocasa.scripts.download_kitchen_assets` in `robocasa` env |
