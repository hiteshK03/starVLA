# /// script
# requires-python = "~=3.11"
# dependencies = [
#     "vla-eval",
#     "starvla",
#     "torch>=2.0",
#     "torchvision>=0.17",
#     "transformers>=4.40,<5",
#     "pillow>=9.0",
#     "opencv-python>=4.0",
#     "numpy>=1.24",
#     "accelerate",
#     "kernels>=0.11.0",
#     "qwen-vl-utils",
#     "omegaconf",
#     "rich",
#     "diffusers",
#     "timm",
#     "einops",
#     "scipy",
#     "transforms3d",
#     "huggingface-hub",
# ]
#
# [tool.uv.sources]
# vla-eval = { path = "../../..", editable = true }
# starvla = { path = "/mnt/dcgpuval/hkandala/starVLA", editable = true }
#
# [tool.uv]
# exclude-newer = "2026-02-24T00:00:00Z"
# ///
"""StarVLA-bridged VLANeXt model server for LIBERO evaluation.

Loads a VLANeXt checkpoint converted to StarVLA's QwenVLANeXt format
via ``convert_vlanext_to_starvla.py`` and runs inference with the
StarVLA framework's ``predict_action()`` method.

Key differences from the generic StarVLA model server:
  - Dense proprioception history (8 frames) accumulated via on_observation()
  - Per-suite action denormalization bounds matching VLANeXt's training
  - quat_no_antipodal for robosuite axis-angle convention
  - Gripper binarization at 0 (not 0.5)
"""

from __future__ import annotations

import contextlib
import logging
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image as PILImage

from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.specs import GRIPPER_CLOSE_POS, IMAGE_RGB, LANGUAGE, RAW, DimSpec
from vla_eval.types import Action, Observation

logger = logging.getLogger(__name__)

# Per-suite action bounds from VLANeXt training (first 6 dims, excluding gripper).
ACTION_BOUNDS: dict[str, tuple[list[float], list[float]]] = {
    "libero_spatial": (
        [-0.9375, -0.9375, -0.9375, -0.1875, -0.3675000071525574, -0.36000001430511475],
        [0.9375, 0.9375, 0.9375, 0.1971428543329239, 0.33642858266830444, 0.375],
    ),
    "libero_object": (
        [-0.8839285969734192, -0.9375, -0.9375, -0.15000000596046448, -0.29035714268684387, -0.32892856001853943],
        [0.9375, 0.8919642567634583, 0.9375, 0.17678570747375488, 0.35035714507102966, 0.1810714304447174],
    ),
    "libero_goal": (
        [-0.9375, -0.9375, -0.9375, -0.2582142949104309, -0.375, -0.2871428430080414],
        [0.9375, 0.9375, 0.9375, 0.3557142913341522, 0.375, 0.375],
    ),
    "libero_10": (
        [-0.9375, -0.9375, -0.9375, -0.23642857372760773, -0.3053571283817291, -0.3675000071525574],
        [0.9375, 0.9375, 0.9375, 0.30000001192092896, 0.29357144236564636, 0.375],
    ),
}


@contextlib.contextmanager
def _block_logging_hijack():
    """Prevent starVLA from clobbering the caller's logging configuration."""
    import logging.config

    _real_dictConfig = logging.config.dictConfig
    logging.config.dictConfig = lambda *_a, **_kw: None
    try:
        yield
    finally:
        logging.config.dictConfig = _real_dictConfig


class StarVLAVLANeXtModelServer(PredictModelServer):
    """StarVLA-bridged VLANeXt model server with dense proprio + per-suite bounds."""

    def __init__(
        self,
        checkpoint: str,
        suite: str = "libero_spatial",
        *,
        num_history: int = 8,
        chunk_size: int = 8,
        action_ensemble: str = "newest",
        **kwargs: Any,
    ) -> None:
        super().__init__(chunk_size=chunk_size, action_ensemble=action_ensemble, **kwargs)
        self.checkpoint = checkpoint
        self.suite = suite
        self.num_history = num_history
        self._model = None
        self._state_histories: dict[str, deque] = {}

        if suite not in ACTION_BOUNDS:
            raise ValueError(f"Unknown suite {suite!r}. Choose from {list(ACTION_BOUNDS)}")
        self._action_min = np.array(ACTION_BOUNDS[suite][0], dtype=np.float32)
        self._action_max = np.array(ACTION_BOUNDS[suite][1], dtype=np.float32)

    def get_observation_params(self) -> dict[str, Any]:
        return {"send_state": True, "send_wrist_image": True, "quat_no_antipodal": True}

    def get_action_spec(self) -> dict[str, DimSpec]:
        return {"gripper": GRIPPER_CLOSE_POS}

    def get_observation_spec(self) -> dict[str, DimSpec]:
        return {"image": IMAGE_RGB, "wrist": IMAGE_RGB, "state": RAW, "language": LANGUAGE}

    async def on_episode_start(self, config: dict[str, Any], ctx: SessionContext) -> None:
        self._state_histories[ctx.session_id] = deque(maxlen=256)
        await super().on_episode_start(config, ctx)

    async def on_episode_end(self, result: dict[str, Any], ctx: SessionContext) -> None:
        self._state_histories.pop(ctx.session_id, None)
        await super().on_episode_end(result, ctx)

    async def on_observation(self, obs: Observation, ctx: SessionContext) -> None:
        """Accumulate dense proprioception from every observation."""
        state = obs.get("states", obs.get("state"))
        if state is not None:
            state_arr = np.asarray(state, dtype=np.float32).flatten()
            if len(state_arr) == 8:
                gripper_scalar = np.clip(
                    1.0 - (np.mean(np.abs(state_arr[6:8])) / 0.04),
                    0.0, 1.0,
                )
                state_arr = np.concatenate([state_arr[:6], [gripper_scalar]])

            sid = ctx.session_id
            if sid not in self._state_histories:
                self._state_histories[sid] = deque(maxlen=256)
            self._state_histories[sid].append(state_arr)
        await super().on_observation(obs, ctx)

    def _load_model(self) -> None:
        if self._model is not None:
            return
        import torch

        path = Path(self.checkpoint)
        if path.is_file() and path.suffix in (".pt", ".safetensors"):
            ckpt_path = str(path)
        else:
            # Try as a directory containing checkpoints/
            ckpt_dir = path / "checkpoints"
            if ckpt_dir.is_dir():
                candidates = sorted(
                    [p for p in ckpt_dir.iterdir() if p.suffix in (".pt", ".safetensors")],
                    key=lambda p: p.name,
                )
                if candidates:
                    ckpt_path = str(candidates[-1])
                else:
                    raise FileNotFoundError(f"No .pt files in {ckpt_dir}")
            else:
                raise FileNotFoundError(f"Cannot resolve checkpoint: {self.checkpoint}")

        with _block_logging_hijack():
            from starVLA.model.framework.base_framework import baseframework

        # Patch Qwen3VL attn_implementation if flash_attn not available
        from transformers import Qwen3VLForConditionalGeneration
        _patches = []
        orig = Qwen3VLForConditionalGeneration.from_pretrained.__func__

        @classmethod
        def _patched(cls, *args, **kwargs):
            if kwargs.get("attn_implementation") == "flash_attention_2":
                try:
                    from transformers.utils import is_flash_attn_2_available
                    if not is_flash_attn_2_available():
                        kwargs["attn_implementation"] = "sdpa"
                except ImportError:
                    kwargs["attn_implementation"] = "sdpa"
            return orig(cls, *args, **kwargs)

        _patches.append((Qwen3VLForConditionalGeneration, "from_pretrained", classmethod(orig)))
        Qwen3VLForConditionalGeneration.from_pretrained = _patched

        try:
            with _block_logging_hijack():
                self._model = baseframework.from_pretrained(ckpt_path)
        finally:
            for obj, attr, orig_val in reversed(_patches):
                setattr(obj, attr, orig_val)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = self._model.to(device).eval()
        logger.info("StarVLA-VLANeXt model loaded on %s from %s", device, ckpt_path)

    def _build_state(self, ctx: SessionContext) -> np.ndarray:
        """Build (num_history, 7) state tensor from accumulated history."""
        history = self._state_histories.get(ctx.session_id, deque())
        states = list(history)[-self.num_history:]
        if not states:
            return np.zeros((self.num_history, 7), dtype=np.float32)
        if len(states) < self.num_history:
            states = [states[0]] * (self.num_history - len(states)) + states
        return np.stack(states)  # (num_history, 7)

    def predict_batch(self, obs_batch: list[Observation], ctx_batch: list[SessionContext]) -> list[Action]:
        self._load_model()
        assert self._model is not None

        examples = []
        for obs, ctx in zip(obs_batch, ctx_batch):
            images_source = obs.get("images", {})
            pil_images = []
            if isinstance(images_source, dict):
                # Ensure agentview comes first, then wrist (matching training order)
                for key in ("agentview", "wrist"):
                    if key in images_source:
                        v = images_source[key]
                        pil_images.append(
                            PILImage.fromarray(v).convert("RGB") if isinstance(v, np.ndarray) else v
                        )
                # Fall back: add any remaining images not already included
                for key, v in images_source.items():
                    if key not in ("agentview", "wrist"):
                        pil_images.append(
                            PILImage.fromarray(v).convert("RGB") if isinstance(v, np.ndarray) else v
                        )
            else:
                pil_images = [
                    PILImage.fromarray(images_source).convert("RGB")
                    if isinstance(images_source, np.ndarray) else images_source
                ]

            example: dict[str, Any] = {
                "image": pil_images,
                "lang": obs.get("task_description", ""),
                "state": self._build_state(ctx),  # (num_history, 7)
            }
            examples.append(example)

        result = self._model.predict_action(examples)
        actions_batch = result["normalized_actions"]  # (B, 8, 7)

        outputs = []
        for i in range(len(obs_batch)):
            raw_actions = np.asarray(actions_batch[i], dtype=np.float32)  # (8, 7)
            actions = raw_actions.copy()
            # Denormalize first 6 dims with per-suite bounds
            actions[:, :6] = (actions[:, :6] + 1) / 2 * (self._action_max - self._action_min) + self._action_min
            # Diffusion head outputs gripper in [-1,1]: negative=close, positive=open.
            # Convert to harness convention (positive=close):
            #   model > 0 → open  → -1.0
            #   model ≤ 0 → close → +1.0
            actions[:, 6] = np.where(raw_actions[:, 6] > 0, -1.0, 1.0)

            # Log first action of chunk for debugging
            ctx = ctx_batch[i]
            step = getattr(ctx, "step", -1)
            if step == 0:
                n_imgs = len(examples[i]["image"])
                logger.info("n_images=%d (expected 2 for multi-view)", n_imgs)
            if step <= 8 or step % 40 == 0:
                logger.info(
                    "step=%d raw=[%s] denorm=[%s]",
                    step,
                    np.array2string(raw_actions[0], precision=4, separator=","),
                    np.array2string(actions[0], precision=4, separator=","),
                )
            outputs.append({"actions": actions})

        return outputs


if __name__ == "__main__":
    from vla_eval.model_servers.serve import run_server

    run_server(StarVLAVLANeXtModelServer)
