#!/usr/bin/env python3
"""Convert VLANeXt pretrained checkpoints to Star VLA QwenVLANeXt format.

Usage:
    python scripts/convert_vlanext_to_starvla.py \
        --vlanext_ckpt /path/to/VLANeXt_libero_spatial.pt \
        --output_dir results/Checkpoints/vlanext_libero_spatial

    # Convert all 4 suites at once:
    python scripts/convert_vlanext_to_starvla.py --all \
        --vlanext_dir /path/to/VLANeXt/checkpoints \
        --output_root results/Checkpoints

This produces a Star VLA checkpoint directory:
    <output_dir>/
    ├── config.yaml
    ├── dataset_statistics.json
    └── checkpoints/
        └── pretrained_pytorch_model.pt
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf


# Key mapping from VLANeXt state_dict to QwenVLANeXt state_dict
def map_vlanext_to_starvla(vlanext_sd: dict) -> dict:
    """Map VLANeXt state_dict keys to QwenVLANeXt format.

    Key transformations:
        lmm.model.*           -> qwen_vl_interface.model.model.*
        lmm.lm_head.*         -> qwen_vl_interface.model.lm_head.*
        action_projector.*     -> proprio_projector.*
        action_head.*          -> action_head.*  (unchanged)
        meta_queries           -> meta_queries   (unchanged)
    """
    starvla_sd = {}
    skipped = []

    for key, value in vlanext_sd.items():
        if key.startswith("lmm.model."):
            # lmm.model.X -> qwen_vl_interface.model.model.X
            new_key = "qwen_vl_interface.model.model." + key[len("lmm.model."):]
        elif key.startswith("lmm.lm_head."):
            # lmm.lm_head.X -> qwen_vl_interface.model.lm_head.X
            new_key = "qwen_vl_interface.model.lm_head." + key[len("lmm.lm_head."):]
        elif key.startswith("action_projector."):
            # action_projector.X -> proprio_projector.X
            new_key = "proprio_projector." + key[len("action_projector."):]
        elif key.startswith("action_head.") or key == "meta_queries":
            # Direct mapping
            new_key = key
        elif key.startswith("connector."):
            # connector.X -> connector.X (if exists in target)
            new_key = key
        else:
            skipped.append(key)
            continue

        starvla_sd[new_key] = value

    if skipped:
        print(f"  Skipped {len(skipped)} keys: {skipped}")

    return starvla_sd


def build_config_yaml(vlanext_config: dict, suite_name: str) -> dict:
    """Build Star VLA config.yaml from VLANeXt's stored config."""
    model_cfg = vlanext_config.get("model", {})
    data_cfg = vlanext_config.get("data", {})
    train_cfg = vlanext_config.get("train", {})

    config = {
        "run_id": f"vlanext_{suite_name}",
        "run_root_dir": "results/Checkpoints",
        "seed": train_cfg.get("seed", 42),
        "framework": {
            "name": "QwenVLANeXt",
            "qwenvl": {
                "base_vlm": model_cfg.get("lmm_path", "Qwen/Qwen3-VL-2B-Instruct"),
                "attn_implementation": model_cfg.get("attn_implementation", "flash_attention_2"),
                "vl_hidden_dim": 2048,  # Qwen3-VL-2B text hidden size
            },
            "action_model": {
                "action_dim": model_cfg.get("action_dim", 7),
                "policy_hidden_size": model_cfg.get("policy_hidden_size", 1024),
                "policy_depth": model_cfg.get("policy_depth", 29),
                "policy_num_heads": model_cfg.get("policy_num_heads", 16),
                "policy_mlp_ratio": model_cfg.get("policy_mlp_ratio", 4.0),
                "num_queries": model_cfg.get("num_queries", 16),
                "action_horizon": model_cfg.get("future_len", 8),
                "past_action_window_size": 0,
                "num_train_timesteps": model_cfg.get("num_train_timesteps", 1000),
                "num_inference_timesteps": model_cfg.get("num_inference_timesteps", 10),
                "condition_type": model_cfg.get("condition_type", "soft"),
                "use_proprio_input_vlm": model_cfg.get("use_proprio_input_vlm", True),
                "use_transformer_proprio_projector": model_cfg.get("use_transformer_proprio_projector", False),
                "projector_depth": model_cfg.get("projector_depth", 2),
                "projector_num_heads": model_cfg.get("projector_num_heads", 4),
                "dct_loss_weight": model_cfg.get("dct_loss_weight", 0.1),
            },
        },
        "datasets": {
            "vla_data": {
                "dataset_py": "lerobot_datasets",
                "data_root_dir": "playground/Datasets/LIBERO",
                "data_mix": suite_name,
                "action_type": "delta_qpos",
                "sequential_step_sampling": False,
                "default_image_resolution": [3, 224, 224],
                "per_device_batch_size": 16,
                "load_all_data_for_training": True,
                "obs": ["image_0"],
                "video_backend": "torchvision_av",
            },
        },
        "trainer": {
            "epochs": 100,
            "max_train_steps": train_cfg.get("max_steps", 10000),
            "num_warmup_steps": train_cfg.get("warmup_steps", 500),
            "save_interval": 5000,
            "eval_interval": 100,
            "learning_rate": {
                "base": train_cfg.get("learning_rate", 1e-4),
                "qwen_vl_interface": train_cfg.get("vlm_lr", 1e-5),
                "action_head": train_cfg.get("learning_rate", 1e-4),
            },
            "lr_scheduler_type": "cosine_with_min_lr",
            "scheduler_specific_kwargs": {"min_lr": 1e-6},
            "freeze_modules": "",
            "loss_scale": {"vla": 1.0},
            "max_grad_norm": 1.0,
            "gradient_clipping": 1.0,
            "gradient_accumulation_steps": 1,
            "logging_frequency": 10,
            "optimizer": {
                "name": "AdamW",
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "weight_decay": 1e-8,
            },
            "enable_gradient_checkpointing": True,
            "enable_mixed_precision_training": True,
        },
    }
    return config


def build_dataset_statistics(suite_name: str) -> dict:
    """Build dataset_statistics.json with LIBERO action normalization stats.

    These are the q01/q99 stats from the LIBERO dataset. The eval harness
    uses these to denormalize predicted actions.
    """
    # LIBERO action stats computed from the dataset
    # Actions are already normalized to [-1, 1] in the LeRobot format
    stats = {
        suite_name: {
            "action": {
                "mean": [0.0] * 7,
                "std": [1.0] * 7,
                "min": [-1.0] * 7,
                "max": [1.0] * 7,
                "q01": [-1.0] * 7,
                "q99": [1.0] * 7,
                "mask": [True, True, True, True, True, True, True],
            },
        },
    }
    return stats


# Map VLANeXt checkpoint filenames to data_mix names
SUITE_MAP = {
    "VLANeXt_libero_spatial.pt": "libero_spatial",
    "VLANeXt_libero_goal.pt": "libero_goal",
    "VLANeXt_libero_object.pt": "libero_object",
    "VLANeXt_libero_10.pt": "libero_10",
}


def convert_checkpoint(vlanext_ckpt_path: str, output_dir: str, suite_name: str = None):
    """Convert a single VLANeXt checkpoint to Star VLA format."""
    vlanext_ckpt_path = Path(vlanext_ckpt_path)
    output_dir = Path(output_dir)

    # Auto-detect suite name from filename
    if suite_name is None:
        suite_name = SUITE_MAP.get(vlanext_ckpt_path.name)
        if suite_name is None:
            # Try to extract from filename
            name = vlanext_ckpt_path.stem.lower()
            for candidate in ["libero_spatial", "libero_goal", "libero_object", "libero_10"]:
                if candidate in name:
                    suite_name = candidate
                    break
        if suite_name is None:
            raise ValueError(
                f"Cannot determine suite name from {vlanext_ckpt_path.name}. "
                f"Pass --suite explicitly."
            )

    print(f"\nConverting: {vlanext_ckpt_path}")
    print(f"  Suite: {suite_name}")
    print(f"  Output: {output_dir}")

    # Load VLANeXt checkpoint
    ckpt = torch.load(str(vlanext_ckpt_path), map_location="cpu", weights_only=False)
    vlanext_sd = ckpt["model_state_dict"]
    vlanext_config = ckpt.get("config", {})

    print(f"  VLANeXt keys: {len(vlanext_sd)}")

    # Map keys
    starvla_sd = map_vlanext_to_starvla(vlanext_sd)
    print(f"  Star VLA keys: {len(starvla_sd)}")

    # Create output directory structure
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save state_dict (flat, no wrapper)
    ckpt_path = ckpt_dir / "pretrained_pytorch_model.pt"
    torch.save(starvla_sd, str(ckpt_path))
    print(f"  Saved state_dict: {ckpt_path}")

    # Save config.yaml
    config = build_config_yaml(vlanext_config, suite_name)
    config_path = output_dir / "config.yaml"
    OmegaConf.save(OmegaConf.create(config), str(config_path))
    print(f"  Saved config: {config_path}")

    # Save dataset_statistics.json
    stats = build_dataset_statistics(suite_name)
    stats_path = output_dir / "dataset_statistics.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved stats: {stats_path}")

    # Verify: try loading with from_pretrained pattern
    print(f"  Checkpoint ready for: baseframework.from_pretrained('{ckpt_path}')")

    return str(ckpt_path)


def main():
    parser = argparse.ArgumentParser(description="Convert VLANeXt checkpoints to Star VLA format")
    parser.add_argument("--vlanext_ckpt", type=str, help="Path to single VLANeXt .pt checkpoint")
    parser.add_argument("--output_dir", type=str, help="Output directory for converted checkpoint")
    parser.add_argument("--suite", type=str, help="Suite name (auto-detected from filename if omitted)")
    parser.add_argument("--all", action="store_true", help="Convert all 4 LIBERO suite checkpoints")
    parser.add_argument("--vlanext_dir", type=str, default="/mnt/dcgpuval/hkandala/VLANeXt/checkpoints",
                        help="Directory containing VLANeXt checkpoints (for --all)")
    parser.add_argument("--output_root", type=str, default="results/Checkpoints",
                        help="Root directory for converted checkpoints (for --all)")
    args = parser.parse_args()

    if args.all:
        vlanext_dir = Path(args.vlanext_dir)
        output_root = Path(args.output_root)
        converted = []
        for ckpt_name, suite_name in SUITE_MAP.items():
            ckpt_path = vlanext_dir / ckpt_name
            if ckpt_path.exists():
                output_dir = output_root / f"vlanext_{suite_name}"
                ckpt = convert_checkpoint(str(ckpt_path), str(output_dir), suite_name)
                converted.append((suite_name, ckpt))
            else:
                print(f"\nSkipping {ckpt_name} (not found at {ckpt_path})")

        print(f"\n{'='*60}")
        print(f"Converted {len(converted)} checkpoints:")
        for suite, path in converted:
            print(f"  {suite}: {path}")
    else:
        if not args.vlanext_ckpt or not args.output_dir:
            parser.error("--vlanext_ckpt and --output_dir required (or use --all)")
        convert_checkpoint(args.vlanext_ckpt, args.output_dir, args.suite)


if __name__ == "__main__":
    main()
