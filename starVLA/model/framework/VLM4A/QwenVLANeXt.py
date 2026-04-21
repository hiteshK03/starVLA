"""QwenVLANeXt: VLANeXt policy architecture on Star VLA training infra.

Registers as "QwenVLANeXt" in FRAMEWORK_REGISTRY.
Uses VLANeXt's MoE diffusion head with per-layer VLM conditioning
instead of Star VLA's GR00T DiT (cross-attention to last hidden state).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_workspace_root = Path(__file__).parent.parent.parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.VLANeXt_ActionHeader import ActionDiffusionTransformerMoE, ActionClassificationTransformerMoE
from starVLA.model.modules.projector.proprio_projector import ActionTransformerProjector, ConnectorTransformer
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

try:
    from deployment.model_server.tools.image_tools import to_pil_preserve
    from starVLA.training.trainer_utils.trainer_tools import resize_images
except ImportError:
    from PIL import Image

    def to_pil_preserve(x):
        if isinstance(x, list):
            return [to_pil_preserve(i) for i in x]
        if isinstance(x, Image.Image):
            return x
        if isinstance(x, np.ndarray):
            return Image.fromarray(x)
        return x

    def resize_images(batch_images, target_size):
        return [[img.resize((target_size[1], target_size[0])) for img in imgs] for imgs in batch_images]


@dataclass
class QwenVLANeXtDefaultConfig:
    name: str = "QwenVLANeXt"
    qwenvl: dict = field(default_factory=lambda: {
        "base_vlm": "Qwen/Qwen3-VL-2B-Instruct",
        "attn_implementation": "flash_attention_2",
        "vl_hidden_dim": 2048,
    })
    action_model: dict = field(default_factory=lambda: {
        "action_dim": 7,
        "policy_hidden_size": 1024,
        "policy_depth": 29,
        "policy_num_heads": 16,
        "policy_mlp_ratio": 4.0,
        "num_queries": 16,
        "action_horizon": 8,
        "past_action_window_size": 0,
        "num_train_timesteps": 1000,
        "num_inference_timesteps": 10,
        # Proprio
        "use_proprio_input_vlm": True,
        "use_transformer_proprio_projector": True,
        "projector_depth": 2,
        "projector_num_heads": 4,
        # Connector (for loose conditioning)
        "use_transformer_connector": True,
        "connector_depth": 2,
        "connector_num_heads": 4,
        # Conditioning: "loose", "tight", "soft"
        "condition_type": "soft",
        # DCT loss
        "dct_loss_weight": 0.1,
        "dct_low_freq_weight": 1.0,
        "dct_high_freq_weight": 3.0,
        "dct_freq_split": 0.5,
        # Classification head
        "loss_type": "diffusion",  # "diffusion" or "classification"
        "num_bins": 256,
    })
    obs_image_size: Optional[list] = None


@FRAMEWORK_REGISTRY.register("QwenVLANeXt")
class Qwen_VLANeXt(baseframework):

    def __init__(self, config=None, **kwargs):
        super().__init__()
        self.config = merge_framework_config(QwenVLANeXtDefaultConfig, config)
        fc = self.config.framework

        # --- VLM ---
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        vlm_hidden_size = self.qwen_vl_interface.model.config.hidden_size

        # Enable gradient flow through backbone to prepended/appended embeddings
        # (required for meta_queries and proprio_projector to receive gradients
        # when gradient checkpointing is active)
        lmm = self.qwen_vl_interface.model
        if hasattr(lmm, "enable_input_require_grads"):
            lmm.enable_input_require_grads()
        if hasattr(lmm, "gradient_checkpointing_enable"):
            lmm.gradient_checkpointing_enable()
        if hasattr(lmm.config, "use_cache"):
            lmm.config.use_cache = False

        # --- Action model config ---
        ac = fc.action_model
        self.action_dim = ac.action_dim
        self.action_horizon = ac.action_horizon
        self.past_action_window_size = ac.get("past_action_window_size", 0)
        self.future_action_window_size = self.action_horizon - 1
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size
        self.num_queries = ac.num_queries
        self.condition_type = ac.condition_type
        self.use_proprio = ac.get("use_proprio_input_vlm", True)
        self.num_train_timesteps = ac.num_train_timesteps
        self.num_inference_timesteps = ac.num_inference_timesteps
        self.dct_loss_weight = ac.get("dct_loss_weight", 0.1)
        self.dct_low_freq_weight = ac.get("dct_low_freq_weight", 1.0)
        self.dct_high_freq_weight = ac.get("dct_high_freq_weight", 3.0)
        self.dct_freq_split = ac.get("dct_freq_split", 0.5)
        self.loss_type = ac.get("loss_type", "diffusion")
        self.num_bins = ac.get("num_bins", 256)

        # --- Meta queries (for "loose" and "soft") ---
        if self.condition_type != "tight":
            self.meta_queries = nn.Parameter(torch.randn(self.num_queries, vlm_hidden_size))

        # --- Proprio projector ---
        if self.use_proprio:
            if ac.get("use_transformer_proprio_projector", True):
                self.proprio_projector = ActionTransformerProjector(
                    action_dim=self.action_dim,
                    hidden_size=vlm_hidden_size,
                    depth=ac.get("projector_depth", 2),
                    num_heads=ac.get("projector_num_heads", 4),
                )
            else:
                self.proprio_projector = nn.Linear(self.action_dim, vlm_hidden_size)

        # --- Connector (for "loose" conditioning) ---
        if self.condition_type == "loose":
            if ac.get("use_transformer_connector", True):
                self.connector = ConnectorTransformer(
                    input_dim=vlm_hidden_size,
                    output_dim=ac.policy_hidden_size,
                    depth=ac.get("connector_depth", 2),
                    num_heads=ac.get("connector_num_heads", 4),
                )
            else:
                self.connector = nn.Sequential(
                    nn.Linear(vlm_hidden_size, ac.policy_hidden_size),
                    nn.SiLU(),
                    nn.Linear(ac.policy_hidden_size, ac.policy_hidden_size),
                )

        # --- Action Head ---
        if self.loss_type == "classification":
            self.action_head = ActionClassificationTransformerMoE(
                action_dim=self.action_dim,
                vlm_hidden_size=vlm_hidden_size,
                num_actions=self.action_horizon,
                num_bins=self.num_bins,
                hidden_size=ac.policy_hidden_size,
                depth=ac.policy_depth,
                num_heads=ac.policy_num_heads,
                mlp_ratio=ac.get("policy_mlp_ratio", 4.0),
            )
        else:
            self.action_head = ActionDiffusionTransformerMoE(
                action_dim=self.action_dim,
                vlm_hidden_size=vlm_hidden_size,
                hidden_size=ac.policy_hidden_size,
                depth=ac.policy_depth,
                num_heads=ac.policy_num_heads,
                mlp_ratio=ac.get("policy_mlp_ratio", 4.0),
            )
            from diffusers import FlowMatchEulerDiscreteScheduler
            self.noise_scheduler = FlowMatchEulerDiscreteScheduler(
                num_train_timesteps=self.num_train_timesteps,
            )

    def _get_vlm_condition(self, qwen_inputs, state=None):
        """Run VLM forward with proprio + meta-query injection.

        Returns:
            connector_out: [B, num_queries, hidden] or None
            hidden_states: tuple of [B, L, H] per VLM layer (for soft/tight)
        """
        B = qwen_inputs["input_ids"].shape[0]
        backbone = self.qwen_vl_interface.model.model  # inner Qwen3VL model
        pad_token_id = getattr(self.qwen_vl_interface.model.config, "pad_token_id", 0) or 0

        input_ids = qwen_inputs["input_ids"]
        attention_mask = qwen_inputs["attention_mask"]
        pixel_values = qwen_inputs.get("pixel_values")
        pixel_values_videos = qwen_inputs.get("pixel_values_videos")
        image_grid_thw = qwen_inputs.get("image_grid_thw")
        video_grid_thw = qwen_inputs.get("video_grid_thw")

        inputs_embeds = backbone.get_input_embeddings()(input_ids)

        # Prepend proprioception — cast to match embedding dtype
        if self.use_proprio and state is not None:
            # Get projector weight dtype (may be bfloat16 under DeepSpeed)
            proj_dtype = next(self.proprio_projector.parameters()).dtype
            state_t = state.to(device=inputs_embeds.device, dtype=proj_dtype)
            proprio_embeds = self.proprio_projector(state_t)
            proprio_embeds = proprio_embeds.to(dtype=inputs_embeds.dtype)
            inputs_embeds = torch.cat([proprio_embeds, inputs_embeds], dim=1)
            proprio_mask = torch.ones(B, state_t.shape[1], device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([proprio_mask, attention_mask], dim=1)
            proprio_ids = torch.full((B, state_t.shape[1]), pad_token_id, dtype=input_ids.dtype, device=input_ids.device)
            input_ids = torch.cat([proprio_ids, input_ids], dim=1)

        # Append meta-queries
        if self.condition_type != "tight":
            queries_embeds = self.meta_queries.unsqueeze(0).expand(B, -1, -1).to(inputs_embeds.dtype)
            inputs_embeds = torch.cat([inputs_embeds, queries_embeds], dim=1)
            queries_mask = torch.ones(B, self.num_queries, device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([attention_mask, queries_mask], dim=1)
            queries_ids = torch.full((B, self.num_queries), pad_token_id, dtype=input_ids.dtype, device=input_ids.device)
            input_ids = torch.cat([input_ids, queries_ids], dim=1)

        # RoPE position ids
        rope_kwargs = {
            "input_ids": input_ids,
            "image_grid_thw": image_grid_thw,
            "video_grid_thw": video_grid_thw,
            "attention_mask": attention_mask,
        }
        position_ids, _ = backbone.get_rope_index(**rope_kwargs)

        output_hidden_states = self.condition_type in ("tight", "soft")
        outputs = backbone(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            output_hidden_states=output_hidden_states,
        )

        hidden_states = outputs.hidden_states if output_hidden_states else None

        connector_out = None
        if self.condition_type == "loose" and hasattr(self, "connector"):
            query_outputs = outputs.last_hidden_state[:, -self.num_queries:, :]
            connector_out = self.connector(query_outputs)

        return connector_out, hidden_states

    def _compute_dct_loss(self, pred, target):
        """DCT auxiliary loss from VLANeXt."""
        B, T, D = pred.shape

        if not hasattr(self, '_dct_matrix') or self._dct_matrix.shape[0] != T or self._dct_matrix.device != pred.device:
            n = torch.arange(T, device=pred.device).float()
            k = torch.arange(T, device=pred.device).float()
            dct_m = torch.cos((np.pi / T) * (n + 0.5).unsqueeze(0) * k.unsqueeze(1))
            dct_m[0, :] *= 1.0 / np.sqrt(T)
            dct_m[1:, :] *= np.sqrt(2.0 / T)
            self._dct_matrix = dct_m

        split_idx = max(1, int(T * self.dct_freq_split))
        freq_weights = torch.ones(T, device=pred.device, dtype=pred.dtype)
        freq_weights[:split_idx] = self.dct_low_freq_weight
        freq_weights[split_idx:] = self.dct_high_freq_weight
        freq_weights = freq_weights.view(1, T, 1)

        pred_dct = torch.matmul(pred.permute(0, 2, 1), self._dct_matrix.t()).permute(0, 2, 1)
        target_dct = torch.matmul(target.permute(0, 2, 1), self._dct_matrix.t()).permute(0, 2, 1)

        diff = (pred_dct - target_dct).abs()
        return (diff * freq_weights).mean()

    def forward(self, examples: List[dict] = None, **kwargs) -> dict:
        batch_images = [ex["image"] for ex in examples]
        instructions = [ex["lang"] for ex in examples]
        actions = [ex["action"] for ex in examples]
        state = [ex["state"] for ex in examples] if "state" in examples[0] else None

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, instructions=instructions,
        )

        # VLM condition
        state_t = None
        if state is not None:
            state_t = torch.tensor(np.array(state), device=qwen_inputs["input_ids"].device,
                                   dtype=torch.float32)
        connector_out, hidden_states = self._get_vlm_condition(qwen_inputs, state=state_t)

        # Action head forward + loss
        actions_t = torch.tensor(np.array(actions),
                                 device=qwen_inputs["input_ids"].device,
                                 dtype=torch.float32)
        # Slice to action_horizon
        actions_target = actions_t[:, -(self.future_action_window_size + 1):, :]
        B = actions_target.shape[0]

        if self.loss_type == "classification":
            # Classification forward — single pass, no noise
            if self.condition_type in ("tight", "soft"):
                pred_logits = self.action_head(hidden_states)
            elif self.condition_type == "loose":
                cond_input = connector_out.mean(dim=1)
                pred_logits = self.action_head(cond_input)
            else:
                raise ValueError(f"Unknown condition type: {self.condition_type}")

            # pred_logits: [B, action_horizon, action_dim, num_bins]
            pose_logits = pred_logits[:, :, :self.action_dim - 1, :]
            gripper_logits = pred_logits[:, :, -1:, :2]

            gt_pose = torch.clamp(actions_target[:, :, :6], -1, 1)
            gt_pose_idx = ((gt_pose + 1) / 2 * (self.num_bins - 1)).round().long()

            gt_gripper = torch.clamp(actions_target[:, :, 6:7], -1, 1)
            gt_gripper_idx = ((gt_gripper + 1) / 2).round().long()

            loss_pose = F.cross_entropy(pose_logits.reshape(-1, self.num_bins), gt_pose_idx.reshape(-1))
            loss_gripper = F.cross_entropy(gripper_logits.reshape(-1, 2), gt_gripper_idx.reshape(-1))
            loss = (loss_pose + loss_gripper) / 2.0

            # DCT loss on continuous predictions from softmax
            loss_dct = torch.tensor(0.0, device=loss.device)
            if self.dct_loss_weight > 0:
                pose_probs = F.softmax(pose_logits.float(), dim=-1)
                bin_centers = torch.linspace(-1, 1, self.num_bins, device=actions_target.device, dtype=pose_probs.dtype)
                pred_pose = torch.sum(pose_probs * bin_centers, dim=-1)

                gripper_probs = F.softmax(gripper_logits.float(), dim=-1)
                pred_gripper = -1.0 + 2.0 * gripper_probs[..., 1]

                pred_continuous = torch.cat([pred_pose, pred_gripper], dim=-1)
                loss_dct = self._compute_dct_loss(pred_continuous, actions_target.float())
                loss = loss + self.dct_loss_weight * loss_dct

            with torch.no_grad():
                gripper_acc = (gripper_logits.squeeze(-2).argmax(-1) == gt_gripper_idx.squeeze(-1)).float().mean()

            return {
                "action_loss": loss,
                "loss_pose": loss_pose.item(),
                "loss_gripper": loss_gripper.item(),
                "loss_dct": loss_dct.item(),
                "gripper_accuracy": gripper_acc.item(),
            }

        # Diffusion forward — flow-matching with noise
        noise = torch.randn_like(actions_target)
        sigmas = torch.rand((B,), device=actions_target.device)
        sigmas_expanded = sigmas.view(B, 1, 1)
        noisy_actions = (1.0 - sigmas_expanded) * actions_target + sigmas_expanded * noise
        noisy_actions = noisy_actions.to(dtype=actions_target.dtype)
        timesteps = sigmas * self.num_train_timesteps
        target = noise - actions_target  # velocity

        if self.condition_type in ("tight", "soft"):
            pred = self.action_head(noisy_actions, timesteps, hidden_states)
        elif self.condition_type == "loose":
            cond_input = connector_out.mean(dim=1)
            pred = self.action_head(noisy_actions, timesteps, cond_input)
        else:
            raise ValueError(f"Unknown condition type: {self.condition_type}")

        loss = F.mse_loss(pred.float(), target.float())

        # DCT auxiliary loss
        if self.dct_loss_weight > 0:
            pred_x_start = noisy_actions - sigmas_expanded * pred
            loss_dct = self._compute_dct_loss(pred_x_start.float(), actions_target.float())
            loss = loss + self.dct_loss_weight * loss_dct

        return {"action_loss": loss}

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs) -> dict:
        if not isinstance(examples, list):
            examples = [examples]
        batch_images = [to_pil_preserve(ex["image"]) for ex in examples]
        instructions = [ex["lang"] for ex in examples]
        state = [ex["state"] for ex in examples] if "state" in examples[0] else None

        train_obs_image_size = getattr(self.config.framework, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, instructions=instructions,
        )

        with torch.autocast("cuda", dtype=torch.bfloat16):
            state_t = None
            if state is not None:
                state_t = torch.from_numpy(np.array(state)).to(
                    qwen_inputs["input_ids"].device, dtype=torch.float32
                )
            connector_out, hidden_states = self._get_vlm_condition(qwen_inputs, state=state_t)

        B = qwen_inputs["input_ids"].shape[0]
        device = qwen_inputs["input_ids"].device

        if self.loss_type == "classification":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if self.condition_type in ("tight", "soft"):
                    logits = self.action_head(hidden_states)
                elif self.condition_type == "loose":
                    cond_input = connector_out.mean(dim=1)
                    logits = self.action_head(cond_input)
                else:
                    raise ValueError(f"Unknown condition type: {self.condition_type}")

            # logits: [B, action_horizon, action_dim, num_bins]
            pose_logits = logits[:, :, :self.action_dim - 1, :]
            gripper_logits = logits[:, :, -1:, :2]

            pose_idx = torch.argmax(pose_logits, dim=-1)
            gripper_idx = torch.argmax(gripper_logits, dim=-1)

            pose_pred = (pose_idx.float() / (self.num_bins - 1)) * 2 - 1
            gripper_pred = gripper_idx.float() * 2 - 1

            action = torch.cat([pose_pred, gripper_pred], dim=-1)
            normalized_actions = action.detach().cpu().float().numpy()
            return {"normalized_actions": normalized_actions}

        # Diffusion inference — iterative denoising
        with torch.autocast("cuda", dtype=torch.float32):
            action = torch.randn(B, self.action_horizon, self.action_dim, device=device)
            action = action.to(dtype=self.qwen_vl_interface.model.dtype)
            self.noise_scheduler.set_timesteps(self.num_inference_timesteps)

            for t in self.noise_scheduler.timesteps:
                timesteps = torch.full((B,), t, device=device)
                if self.condition_type in ("tight", "soft"):
                    output = self.action_head(action, timesteps, hidden_states)
                elif self.condition_type == "loose":
                    cond_input = connector_out.mean(dim=1)
                    output = self.action_head(action, timesteps, cond_input)
                else:
                    raise ValueError(f"Unknown condition type: {self.condition_type}")
                action = self.noise_scheduler.step(output, t, action).prev_sample
                action = action.to(dtype=self.qwen_vl_interface.model.dtype)

        normalized_actions = action.detach().cpu().float().numpy()
        return {"normalized_actions": normalized_actions}
