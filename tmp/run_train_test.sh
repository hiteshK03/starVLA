#!/bin/bash
set -e

# Setup conda
source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate starVLA

# Find real CUDA — this cluster doesn't have system nvcc, use conda's cuda-toolkit
# or set CUDA_HOME to the conda env which has CUDA libs
CONDA_CUDA_DIR="$CONDA_PREFIX"
if [ -f "$CONDA_PREFIX/bin/ptxas" ]; then
    export CUDA_HOME="$CONDA_PREFIX"
elif [ -d "$CONDA_PREFIX/pkgs" ]; then
    # Look for cuda-toolkit in conda
    CUDA_TOOLKIT=$(find "$CONDA_PREFIX" -path "*/targets/x86_64-linux/bin/nvcc" 2>/dev/null | head -1)
    if [ -n "$CUDA_TOOLKIT" ]; then
        export CUDA_HOME=$(dirname $(dirname $(dirname "$CUDA_TOOLKIT")))
    fi
fi

# Fallback: create a nvcc wrapper that DeepSpeed can use
if ! command -v nvcc &>/dev/null || ! nvcc --version 2>&1 | grep -q "release"; then
    WRAPPER_DIR="${CONDA_PREFIX}/cuda_compat/bin"
    mkdir -p "$WRAPPER_DIR" 2>/dev/null || true
    TORCH_CUDA_VER=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "12.4")
    MAJOR=$(echo $TORCH_CUDA_VER | cut -d. -f1)
    MINOR=$(echo $TORCH_CUDA_VER | cut -d. -f2)
    cat > "$WRAPPER_DIR/nvcc" << NVCC_EOF
#!/bin/bash
echo "nvcc: NVIDIA (R) Cuda compiler driver"
echo "Cuda compilation tools, release ${MAJOR}.${MINOR}, V${TORCH_CUDA_VER}"
NVCC_EOF
    chmod +x "$WRAPPER_DIR/nvcc"
    export PATH="$WRAPPER_DIR:$PATH"
    export CUDA_HOME="$CONDA_PREFIX/cuda_compat"
    echo "[INFO] Created nvcc wrapper: CUDA ${TORCH_CUDA_VER}"
fi

echo "[INFO] CUDA_HOME=$CUDA_HOME"
echo "[INFO] nvcc=$(which nvcc)"
nvcc --version

cd /home/jye624/Projcets/starVLA

export WANDB_MODE=disabled
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000
export TRITON_CACHE_DIR="/tmp/${USER}_triton_cache"
mkdir -p "${TRITON_CACHE_DIR}" 2>/dev/null || true

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 4 \
  starVLA/training/train_starvla.py \
  --config_yaml ./examples/LIBERO/train_files/starvla_cotrain_libero.yaml \
  --framework.name QwenOFT \
  --framework.qwenvl.base_vlm /home/jye624/Models/Pretrained_models/Qwen3-VL-4B-Instruct \
  --framework.action_model.future_action_window_size 7 \
  --framework.action_model.past_action_window_size 0 \
  --datasets.vla_data.data_root_dir /home/jye624/Datasets/LIBERO \
  --datasets.vla_data.data_mix libero_all \
  --datasets.vla_data.per_device_batch_size 8 \
  --trainer.vla_data.video_backend torchvision_av \
  --framework.qwenvl.attn_implementation sdpa \
  --trainer.freeze_modules '' \
  --trainer.max_train_steps 100 \
  --trainer.save_interval 50 \
  --trainer.logging_frequency 10 \
  --trainer.eval_interval 100 \
  --run_root_dir ./results/Checkpoints \
  --run_id test_config_fix \
  --wandb_project starVLA_Libero \
  --wandb_entity jinhuiye
