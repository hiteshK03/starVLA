# 🛠️ Installation Guide: RoboCasa GR-1 + StarVLA

This guide walks you through setting up **two separate conda environments**:

| Environment | Purpose |
|-------------|---------|
| `starVLA` | Model training & policy server (inference) |
| `robocasa` | RoboCasa-GR1 simulation & evaluation |

> **Prerequisites**: NVIDIA GPU (verified on A100), CUDA ≥ 12.1, conda installed.

---

## 1. StarVLA Environment

If you have already set up the `starVLA` environment (e.g., following the LIBERO guide), you can skip this section.  
Otherwise, install it from the project root:

```bash
bash install_starvla.sh
conda activate starVLA
```

---

## 2. RoboCasa Environment

### 2.1 Create a new conda environment

```bash
conda create -n robocasa python=3.10 -y
conda activate robocasa
```

### 2.2 Install robosuite (required by robocasa)

```bash
pip install robosuite
```

### 2.3 Install robocasa-gr1-tabletop-tasks

Follow the [official installation guide](https://github.com/robocasa/robocasa-gr1-tabletop-tasks?tab=readme-ov-file#getting-started):

```bash
git clone https://github.com/robocasa/robocasa-gr1-tabletop-tasks.git
cd robocasa-gr1-tabletop-tasks
pip install -e .
```

Then download the robocasa assets:

```bash
python -m robocasa.scripts.download_kitchen_assets
```

### 2.4 Install additional dependencies for StarVLA evaluation client

```bash
pip install tyro
pip install websockets opencv-python matplotlib
```

> **Note**: The evaluation client in `examples/Robocasa_tabletop/eval_files/` communicates with the StarVLA policy server via WebSocket, so no deep-learning packages are required in the `robocasa` environment.

---

## 3. Download Pre-trained Model Checkpoints

Activate the `starVLA` environment and download the checkpoints from HuggingFace:

```bash
conda activate starVLA

# Option A: GR00T backbone
huggingface-cli download StarVLA/Qwen3-VL-GR00T-Robocasa-gr1 \
    --repo-type model \
    --local-dir ./playground/Pretrained_models/StarVLA/Qwen3-VL-GR00T-Robocasa-gr1

# Option B: OFT backbone (used in batch_eval_args.sh by default)
huggingface-cli download StarVLA/Qwen3-VL-OFT-Robocasa \
    --repo-type model \
    --local-dir ./playground/Pretrained_models/StarVLA/Qwen3-VL-OFT-Robocasa
```

You also need the base VLM backbone for training:

```bash
huggingface-cli download Qwen/Qwen3-VL-4B-Instruct \
    --repo-type model \
    --local-dir ./playground/Pretrained_models/Qwen3-VL-4B-Instruct
```

---

## 4. Download Training Dataset

The training data comes from the NVIDIA [PhysicalAI-Robotics-GR00T-X-Embodiment-Sim](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim) dataset on HuggingFace.

We only need the `*_1000` fine-tuning folders (24 tasks). A download script is provided:

```bash
conda activate starVLA

# Recommended: use hf-mirror.com to avoid connection resets on the large repo
export HF_ENDPOINT=https://hf-mirror.com

# Run in background (download is large, several hundred GB)
nohup python examples/Robocasa_tabletop/train_files/download_gr00t_ft_data.py \
    > download_gr00t.log 2>&1 &
echo "Download PID: $!"

# Monitor progress
tail -f download_gr00t.log
```

> **Note**: The download script uses the mirror's **per-folder tree API** to list files, then downloads each file individually via `hf_hub_download`. This avoids the connection resets that occur when listing the entire large repo at once. Each folder is retried up to 20 times on failure.

After downloading, verify the directory structure:

```
playground/Datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/
├── gr1_unified.PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_1000/
│   ├── data/chunk-000/episode_XXXXXX.parquet   # action/state trajectories
│   ├── meta/info.json                           # dataset metadata
│   └── videos/chunk-000/observation.images.ego_view/episode_XXXXXX.mp4
├── gr1_unified.PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_1000/
├── ...  (24 folders total)
└── gr1_unified.PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_1000/
```

---

## 5. Visualise the Dataset (Optional)

A visualisation script is provided to inspect downloaded episodes:

```bash
conda activate starVLA

# Print episode summary (state/action dims, task description, success)
python examples/Robocasa_tabletop/visualize_dataset.py \
    --dataset_dir playground/Datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/gr1_unified.PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_1000 \
    --episode_index 0 \
    --no_plot

# Generate a video with action/state overlay (requires videos/ to be downloaded)
python examples/Robocasa_tabletop/visualize_dataset.py \
    --dataset_dir playground/Datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/gr1_unified.PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_1000 \
    --episode_index 0 \
    --save_video episode_0_viz.mp4

# Show matplotlib state/action curves (interactive)
python examples/Robocasa_tabletop/visualize_dataset.py \
    --dataset_dir playground/Datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/gr1_unified.PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_1000 \
    --episode_index 0
```

Example output:
```
Steps: 376  |  Duration: 18.75s  |  State dim: 44  |  Action dim: 44
Task: unlocked_waist: pick up the bottled water, place it into the cabinet and close the cabinet
Success: True
```

---

## 5. Directory Layout Summary

After completing all steps, your workspace should look like:

```
starVLA/
├── playground/
│   ├── Pretrained_models/
│   │   ├── Qwen3-VL-4B-Instruct/          # base VLM for training
│   │   └── StarVLA/
│   │       ├── Qwen3-VL-GR00T-Robocasa-gr1/   # eval checkpoint (GR00T)
│   │       └── Qwen3-VL-OFT-Robocasa/         # eval checkpoint (OFT)
│   └── Datasets/
│       └── nvidia/
│           └── PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/
│               └── gr1_unified.*_1000/    # 24 task folders
├── examples/Robocasa_tabletop/
│   ├── train_files/
│   └── eval_files/
└── deployment/model_server/
```

---

## 6. Verify Installation

### Check StarVLA environment

```bash
conda activate starVLA
python -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
python -c "import starVLA; print('starVLA OK')"
```

### Check RoboCasa environment

```bash
conda activate robocasa
python -c "import robosuite; print('robosuite OK')"
python -c "import robocasa; print('robocasa OK')"
python -c "import tyro; print('tyro OK')"
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: robocasa` | Make sure you ran `pip install -e .` inside the `robocasa-gr1-tabletop-tasks` repo with the `robocasa` conda env active |
| `ModuleNotFoundError: tyro` | Run `pip install tyro` in the `robocasa` env |
| HuggingFace download timeout | Set `HF_ENDPOINT=https://hf-mirror.com` before downloading, or use a VPN |
| CUDA out of memory during training | Reduce `per_device_batch_size` in the training script; see the Getting Started guide |
| `robocasa assets not found` | Run `python -m robocasa.scripts.download_kitchen_assets` in the `robocasa` env |
