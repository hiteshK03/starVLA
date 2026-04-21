# Copyright 2025 NVIDIA Corp. and affiliates. All rights reserved.
# Modified by [Fangjing Wang/ SUST University] in [2025].
# Modification: [return raw data and suport multi-dataset mixture].
# Modified by [Jinhui YE/ HKUST University] in [2025].
# Modification: [suport topdowm processing, suport param from config].

from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import torchvision.transforms.functional as TVF

from starVLA.dataloader.gr00t_lerobot.registry import (
    ROBOT_TYPE_CONFIG_MAP,
    ROBOT_TYPE_TO_EMBODIMENT_TAG,
    DATASET_NAMED_MIXTURES,
)
from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotMixtureDataset, LeRobotSingleDataset
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag


def collate_fn(batch):
    return batch


def make_augmenting_collate_fn(base_collate_fn, aug_cfg):
    """Wrap a collate_fn to apply data augmentation to PIL images.

    Mirrors VLANeXt's DataCollatorForVLANeXt._augment_frames_uint8 exactly:
    uses RandomResizedCrop.get_params() for crop sampling, np.random for
    color jitter, configurable augment_order, and hue clipping.

    Args:
        base_collate_fn: The underlying collate function (identity or padding).
        aug_cfg: OmegaConf node with augmentation parameters.
    """
    from torchvision.transforms import RandomResizedCrop

    rrc_cfg = aug_cfg.get("random_resized_crop", None)
    rrc_scale = tuple(rrc_cfg.scale) if rrc_cfg else (0.9, 0.9)
    rrc_ratio = tuple(rrc_cfg.ratio) if rrc_cfg else (1.0, 1.0)

    rb = list(aug_cfg.get("random_brightness", [])) or None
    rc = list(aug_cfg.get("random_contrast", [])) or None
    rs = list(aug_cfg.get("random_saturation", [])) or None
    rh = list(aug_cfg.get("random_hue", [])) or None
    augment_order = list(aug_cfg.get("augment_order", [
        "random_resized_crop", "random_brightness",
        "random_contrast", "random_saturation", "random_hue",
    ]))

    def _uniform(a, b):
        return float(np.random.uniform(a, b))

    def _sample_brightness():
        if not rb:
            return 1.0
        if len(rb) == 1:
            x = float(rb[0])
            return _uniform(1.0 - x, 1.0 + x)
        return _uniform(float(rb[0]), float(rb[1]))

    def _sample_contrast():
        if not rc:
            return 1.0
        if len(rc) == 1:
            x = float(rc[0])
            return _uniform(1.0 - x, 1.0 + x)
        return _uniform(float(rc[0]), float(rc[1]))

    def _sample_saturation():
        if not rs:
            return 1.0
        if len(rs) == 1:
            x = float(rs[0])
            return _uniform(1.0 - x, 1.0 + x)
        return _uniform(float(rs[0]), float(rs[1]))

    def _sample_hue():
        if not rh:
            return 0.0
        if len(rh) == 1:
            x = float(rh[0])
            return _uniform(-x, x)
        return _uniform(float(rh[0]), float(rh[1]))

    def _augment_images(images):
        """Apply augmentation to a list of PIL images with shared random params."""
        if not images or not augment_order:
            return images

        out_h, out_w = images[0].size[1], images[0].size[0]  # PIL: (w, h)

        # Sample crop params using torchvision's rejection sampling (same as VLANeXt)
        crop_params = None
        if "random_resized_crop" in augment_order and rrc_cfg is not None:
            i, j, h, w = RandomResizedCrop.get_params(images[0], scale=rrc_scale, ratio=rrc_ratio)
            crop_params = (i, j, h, w)

        # Sample color jitter params once per sample
        b_fac = _sample_brightness() if "random_brightness" in augment_order else 1.0
        c_fac = _sample_contrast() if "random_contrast" in augment_order else 1.0
        s_fac = _sample_saturation() if "random_saturation" in augment_order else 1.0
        h_del = _sample_hue() if "random_hue" in augment_order else 0.0
        h_del = float(np.clip(h_del, -0.5, 0.5))

        augmented = []
        for img in images:
            for op in augment_order:
                if op == "random_resized_crop" and crop_params is not None:
                    ci, cj, ch, cw = crop_params
                    img = TVF.resized_crop(img, ci, cj, ch, cw, size=(out_h, out_w))
                elif op == "random_brightness":
                    img = TVF.adjust_brightness(img, b_fac)
                elif op == "random_contrast":
                    img = TVF.adjust_contrast(img, c_fac)
                elif op == "random_saturation":
                    img = TVF.adjust_saturation(img, s_fac)
                elif op == "random_hue":
                    img = TVF.adjust_hue(img, h_del)
            augmented.append(img)
        return augmented

    def augmenting_collate_fn(batch):
        for sample in batch:
            if "image" in sample and isinstance(sample["image"], list):
                sample["image"] = _augment_images(sample["image"])
        return base_collate_fn(batch)

    return augmenting_collate_fn


def make_padding_collate_fn(action_dim: int, action_horizon: int, state_dim: int | None = None):
    """Create a collate_fn that pads action (and optionally state) to uniform dimensions.

    Pads with zeros on the dim axis (right) and chunk/time axis (end).
    Raises ValueError if the source dimensions exceed the target dimensions.

    Args:
        action_dim: Target action dimension (second axis).
        action_horizon: Target action chunk length (first axis).
        state_dim: Target state dimension. If None, state is not padded.
    """

    def _pad_array(arr: np.ndarray, target_time: int, target_dim: int, name: str) -> np.ndarray:
        """Pad a [T, D] array to [target_time, target_dim] with zeros."""
        t, d = arr.shape
        if d > target_dim:
            raise ValueError(
                f"{name} dim ({d}) exceeds target dim ({target_dim}). "
                f"Check your config or dataset — source data should not be wider than the target."
            )
        if t > target_time:
            raise ValueError(
                f"{name} chunk length ({t}) exceeds target chunk length ({target_time}). "
                f"Check your config or dataset — source data should not be longer than the target."
            )
        if t == target_time and d == target_dim:
            return arr
        padded = np.zeros((target_time, target_dim), dtype=arr.dtype)
        padded[:t, :d] = arr
        return padded

    def padding_collate_fn(batch):
        for sample in batch:
            if "action" in sample:
                sample["action"] = _pad_array(sample["action"], action_horizon, action_dim, "action")
            if state_dim is not None and "state" in sample:
                state_time = sample["state"].shape[0]  # keep original time dim for state
                sample["state"] = _pad_array(sample["state"], state_time, state_dim, "state")
        return batch

    return padding_collate_fn


def make_LeRobotSingleDataset(
    data_root_dir: Path | str,
    data_name: str,
    robot_type: str,
    delete_pause_frame: bool = False,
    data_cfg: dict | None = None,
    lerobot_version: str | None = None,
) -> LeRobotSingleDataset:
    """
    Make a LeRobotSingleDataset object.

    :param data_root_dir: The root directory of the dataset.
    :param data_name: The name of the dataset.
    :param robot_type: The robot type config to use.
    :param lerobot_version: Explicit lerobot version override ("v2.0" or "v3.0"). If None, auto-detected from dataset file structure.
    :return: A LeRobotSingleDataset object.
    """

    data_config = ROBOT_TYPE_CONFIG_MAP[robot_type]
    modality_config = data_config.modality_config()
    transforms = data_config.transform()
    dataset_path = data_root_dir / data_name
    if robot_type not in ROBOT_TYPE_TO_EMBODIMENT_TAG:
        print(
            f"Warning: Robot type {robot_type} not found in ROBOT_TYPE_TO_EMBODIMENT_TAG, using {EmbodimentTag.NEW_EMBODIMENT} as default"
        )
        embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    else:
        embodiment_tag = ROBOT_TYPE_TO_EMBODIMENT_TAG[robot_type]

    video_backend = data_cfg.get("video_backend", "decord") if data_cfg else "torchvision_av"
    return LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=modality_config,
        transforms=transforms,
        embodiment_tag=embodiment_tag,
        video_backend=video_backend,  # decord is more efficiency | torchvision_av for video.av1
        delete_pause_frame=delete_pause_frame,
        data_cfg=data_cfg,
        lerobot_version=lerobot_version,
    )


def get_vla_dataset(
    data_cfg: dict,
    mode: str = "train",
    balance_dataset_weights: bool = False,
    balance_trajectory_weights: bool = False,
    seed: int = 42,
    **kwargs: dict,
) -> LeRobotMixtureDataset:
    """
    Get a LeRobotMixtureDataset object.
    """
    data_root_dir = data_cfg.data_root_dir
    data_mix = data_cfg.data_mix
    delete_pause_frame = data_cfg.get("delete_pause_frame", False)
    mixture_spec = DATASET_NAMED_MIXTURES[data_mix]
    included_datasets, filtered_mixture_spec = set(), []
    for d_name, d_weight, robot_type in mixture_spec:
        dataset_key = (d_name, robot_type)
        if dataset_key in included_datasets:
            print(f"Skipping Duplicate Dataset: `{(d_name, d_weight, robot_type)}`")
            continue

        included_datasets.add(dataset_key)
        filtered_mixture_spec.append((d_name, d_weight, robot_type))

    dataset_mixture = []
    for d_name, d_weight, robot_type in filtered_mixture_spec:
        dataset_mixture.append(
            (
                make_LeRobotSingleDataset(
                    Path(data_root_dir), d_name, robot_type, delete_pause_frame=delete_pause_frame, data_cfg=data_cfg
                ),
                d_weight,
            )
        )

    return LeRobotMixtureDataset(
        dataset_mixture,
        mode=mode,
        balance_dataset_weights=balance_dataset_weights,
        balance_trajectory_weights=balance_trajectory_weights,
        seed=seed,
        data_cfg=data_cfg,
        **kwargs,
    )


if __name__ == "__main__":

    # import debugpy
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="./examples/LIBERO/train_files/starvla_cotrain_libero.yaml",
        help="Path to YAML config",
    )
    parser.add_argument(
        "--data_mix",
        type=str,
        default=None,
        help="Override data_mix in config (e.g. libero_goal)",
    )
    args, clipargs = parser.parse_known_args()

    # debugpy.listen(("0.0.0.0", 10092))
    # print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    # debugpy.wait_for_client()
    args.config_yaml = args.config_yaml  # use CLI arg or default
    cfg = OmegaConf.load(args.config_yaml)
    vla_dataset_cfg = cfg.datasets.vla_data
    if hasattr(args, 'data_mix') and args.data_mix:
        vla_dataset_cfg.data_mix = args.data_mix
    vla_dataset_cfg.task_id = "all"
    print(f"Config: {args.config_yaml}")
    print(f"Data mix: {vla_dataset_cfg.data_mix}")
    dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)
        # dataset
    from torch.utils.data import DataLoader

    train_dataloader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=1,  # For Debug
        collate_fn=collate_fn,
    )

    cfg.output_dir = "./results/debug"
    output_dir = Path(cfg.output_dir)
    dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")

    from tqdm import tqdm

    count = 0
    for batch in tqdm(train_dataloader, desc="Processing Batches"):
        # print(batch)
        # print(1)
        if count > 100:
            break
        count += 1
        pass
