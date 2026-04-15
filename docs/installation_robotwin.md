# Installation Guide: RoboTwin 2.0

This guide documents the complete environment setup for running RoboTwin 2.0 simulation and training with StarVLA.

For the base StarVLA installation (starvla env), see [installation.md](installation.md).

---

## Prerequisites

- Python 3.10
- NVIDIA GPU with CUDA 12.x driver (IsaacSim requires a compatible GPU)
- Linux (Ubuntu 20.04+)
- Miniconda installed at `~/miniconda3`
- `starvla` conda env already set up (see [installation.md](installation.md))

---

## Step 1: Install RoboTwin Base Environment

Follow the [official RoboTwin installation guide](https://robotwin-platform.github.io/doc/usage/robotwin-install.html) to create the base `robotwin` conda environment with IsaacSim/IsaacGym support.

> [!IMPORTANT]
> The RoboTwin base environment requires NVIDIA IsaacSim. Make sure your GPU driver is compatible before proceeding.

After completing the official guide, you should have a `robotwin` conda environment.

---

## Step 2: Install StarVLA Dependencies in `starvla` Env

```bash
cd starVLA
~/miniconda3/envs/starvla/bin/pip install -r requirements.txt
```

---

## Step 3: Install RoboTwin Eval-Side Dependencies in `robotwin` Env

```bash
cd starVLA
~/miniconda3/envs/robotwin/bin/pip install -r examples/Robotwin/eval_files/requirements.txt
```

Or use the provided install script (handles both steps 2 and 3 automatically):

```bash
cd starVLA
bash examples/Robotwin/eval_files/install_robotwin_env.sh
```

---

## Step 4: Download RoboTwin 2.0 Clean Dataset

The Clean dataset contains 50 tasks × 50 trajectories per task.

```bash
cd starVLA
mkdir -p playground/Datasets/RoboTwin

# Download Clean subset (via hf-mirror.com for mainland China)
HF_ENDPOINT=https://hf-mirror.com \
  ~/miniconda3/envs/starvla/bin/huggingface-cli download \
  RoboTwin/RoboTwin2.0-Dataset \
  --repo-type dataset \
  --local-dir playground/Datasets/RoboTwin \
  --include "Clean/*" \
  --resume-download
```

Or use the provided download script:

```bash
cd starVLA
bash examples/Robotwin/train_files/download_robotwin_clean.sh
```

> [!TIP]
> To also download the Randomized dataset (50 tasks × 500 trajectories), remove the `--include "Clean/*"` filter or add `--include "Randomized/*"`.

---

## Step 5: Convert HDF5 Dataset to LeRobot Format

The RoboTwin 2.0 dataset is distributed as HDF5 files. StarVLA's dataloader requires **LeRobot v2.1 format** (parquet + mp4 + `meta/info.json`). Use the provided conversion script:

```bash
cd starVLA

# Convert a single task (e.g., adjust_bottle)
~/miniconda3/envs/starvla/bin/python \
  examples/Robotwin/train_files/convert_robotwin_to_lerobot.py \
  --input_dir playground/Datasets/RoboTwin/Clean/adjust_bottle/aloha-agilex_clean_50 \
  --task_name adjust_bottle \
  --fps 25

# Convert all downloaded tasks at once
~/miniconda3/envs/starvla/bin/python \
  examples/Robotwin/train_files/convert_robotwin_to_lerobot.py \
  --all \
  --data_root playground/Datasets/RoboTwin/Clean \
  --fps 25
```

After conversion, each task directory will contain:
```
playground/Datasets/RoboTwin/Clean/<task>/aloha-agilex_clean_50/
├── data/chunk-000/
│   ├── episode_000000.parquet
│   ├── episode_000001.parquet
│   └── ...
├── videos/chunk-000/
│   ├── observation.images.cam_high/episode_000000.mp4
│   ├── observation.images.cam_left_wrist/episode_000000.mp4
│   └── observation.images.cam_right_wrist/episode_000000.mp4
└── meta/
    ├── info.json        ← required by dataloader
    ├── modality.json    ← action/state field mapping
    ├── tasks.jsonl
    └── episodes.jsonl
```

> [!NOTE]
> The conversion script requires `opencv-python` (`cv2`) and `h5py`, both of which are available in the `starvla` conda environment after running `pip install -r requirements.txt`.

---

## Step 6: Clone RoboTwin Repository and Apply Patch

The evaluation launcher requires a local RoboTwin checkout with a small patch applied to `script/eval_policy.py`.

### 6.1 Clone RoboTwin

```bash
git clone https://github.com/RoboTwin-Platform/RoboTwin.git /path/to/RoboTwin
export ROBOTWIN_PATH=/path/to/RoboTwin
```

### 6.2 Apply the `eval_policy.py` Patch

Apply the following change to your local RoboTwin checkout so that `script/eval_policy.py` accepts `--policy_ckpt_path`:

```diff
diff --git a/script/eval_policy.py b/script/eval_policy.py
index eded198..9fb36e3 100644
--- a/script/eval_policy.py
+++ b/script/eval_policy.py
@@ -69,6 +69,7 @@ def main(usr_args):
     policy_name = usr_args["policy_name"]
     instruction_type = usr_args["instruction_type"]
+    policy_ckpt_path = usr_args["policy_ckpt_path"]
     save_dir = None

@@ -81,6 +82,7 @@ def main(usr_args):
     args['task_name'] = task_name
     args["task_config"] = task_config
     args["ckpt_setting"] = ckpt_setting
+    args["policy_ckpt_path"] = policy_ckpt_path

@@ -327,11 +329,13 @@ def parse_args_and_config():
     parser = argparse.ArgumentParser()
     parser.add_argument("--config", type=str, required=True)
+    parser.add_argument("--policy_ckpt_path", type=str, required=True)
     parser.add_argument("--overrides", nargs=argparse.REMAINDER)
     args = parser.parse_args()

     with open(args.config, "r", encoding="utf-8") as f:
         config = yaml.safe_load(f)
+    config["policy_ckpt_path"] = args.policy_ckpt_path
```

Apply with:

```bash
cd /path/to/RoboTwin
patch -p1 < /path/to/eval_policy.patch
```

Or apply manually by editing `script/eval_policy.py` as shown above.

> [!NOTE]
> This patch is intentionally not vendored into the StarVLA repo because RoboTwin is maintained in a separate repository. The StarVLA launcher passes `--policy_ckpt_path` at runtime; without this patch, RoboTwin cannot forward the checkpoint path into `model2robotwin_interface.py`.

---

## Step 7: Set Environment Variables

Add the following to your `~/.bashrc` or set them before running eval:

```bash
# Path to your local RoboTwin checkout
export ROBOTWIN_PATH=/path/to/RoboTwin

# (Optional) Explicit Python binary paths — auto-detected if unset
export STARVLA_PYTHON=~/miniconda3/envs/starvla/bin/python
export ROBOTWIN_PYTHON=~/miniconda3/envs/robotwin/bin/python
```

---

## Step 8: Verify Installation

```bash
# Verify robotwin env dependencies
~/miniconda3/envs/robotwin/bin/python -c "
import accelerate, websockets, msgpack, rich, omegaconf
print('robotwin env OK')
print('  accelerate:', accelerate.__version__)
print('  websockets:', websockets.__version__)
"

# Verify starvla env
cd starVLA
~/miniconda3/envs/starvla/bin/python -c "
import torch, flash_attn, starVLA
print('starvla env OK')
print('torch:', torch.__version__, '| cuda:', torch.cuda.is_available())
"

# Verify dataset structure
ls playground/Datasets/RoboTwin/Clean/ | head -5
ls playground/Datasets/RoboTwin/Clean/adjust_bottle/meta/modality.json
```

Expected output:
```
robotwin env OK
  accelerate: 1.5.2
  websockets: 15.0.1
starvla env OK
torch: 2.6.0+cu124 | cuda: True
adjust_bottle
...
playground/Datasets/RoboTwin/Clean/adjust_bottle/meta/modality.json
```

---

## Known Issues & Fixes

### `ROBOTWIN_PATH` not set
The eval launcher requires `ROBOTWIN_PATH` to point to your local RoboTwin checkout.
Fix: `export ROBOTWIN_PATH=/path/to/RoboTwin`

### `policy_ckpt_path` KeyError in RoboTwin eval
The `eval_policy.py` patch (Step 6) has not been applied.
Fix: Apply the patch as described in Step 6.

### HuggingFace download timeout
Use `--resume-download` flag (already included in the download script) or download individual files with `wget -c` from `hf-mirror.com`.

### `CUDA_HOME does not exist` (DeepSpeed during training)
Fix: `export CUDA_HOME=~/miniconda3/envs/starvla`

### IsaacSim GPU compatibility
RoboTwin simulation requires a compatible NVIDIA GPU. Check the [official RoboTwin docs](https://robotwin-platform.github.io/doc/usage/robotwin-install.html) for supported GPU models.
