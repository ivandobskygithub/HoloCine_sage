import glob
import logging
import math
import os
import sys
import types
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from einops import rearrange, reduce, repeat
from modelscope import snapshot_download
from tqdm import tqdm
from typing_extensions import Literal

from ..utils import BasePipeline, ModelConfig, PipelineUnit, PipelineUnitRunner
from ..models import ModelManager, load_state_dict
from ..models.wan_video_dit import WanModel, RMSNorm, sinusoidal_embedding_1d
from ..models.wan_video_text_encoder import WanTextEncoder, T5RelativeEmbedding, T5LayerNorm
from ..models.wan_video_vae import WanVideoVAE, RMS_norm, CausalConv3d, Upsample
from ..models.wan_video_image_encoder import WanImageEncoder
from ..models.wan_video_vace import VaceWanModel
from ..models.wan_video_motion_controller import WanMotionControllerModel
from ..schedulers.flow_match import FlowMatchScheduler
from ..prompters import WanPrompter
from ..vram_management import enable_vram_management, AutoWrappedModule, AutoWrappedLinear, WanAutoCastLayerNorm
from ..lora import GeneralLoRALoader


BLOCK_SWAP_LOGGER_NAME = "holocine.blockswap"


@dataclass
class BlockSwapConfig:
    offload_device: torch.device
    offload_dtype: torch.dtype
    sliding_window_size: int
    sliding_window_stride: int
    limit_gb: float


@dataclass
class GPUMemorySnapshot:
    total_bytes: int
    free_bytes: int
    allocated_bytes: int

    @property
    def total_gb(self) -> float:
        return self.total_bytes / (1024 ** 3)

    @property
    def free_gb(self) -> float:
        return self.free_bytes / (1024 ** 3)

    @property
    def allocated_gb(self) -> float:
        return self.allocated_bytes / (1024 ** 3)


@dataclass
class BlockSwapPlan:
    use_block_swap: bool
    config: Optional[BlockSwapConfig]
    storage_device: torch.device
    storage_dtype: torch.dtype
    available_gb: Optional[float]
    total_latent_gb: float
    window_latent_gb: float
    model_gb: float
    window_size: int
    window_stride: int
    effective_limit_gb: float
    offload_models: bool
    vram_limit_gb: Optional[float]
    reason: str



class WanVideoHoloCinePipeline(BasePipeline):

    _LOGGER_NAME = BLOCK_SWAP_LOGGER_NAME

    def _configure_blockswap_logger(self) -> logging.Logger:
        logger = logging.getLogger(self._LOGGER_NAME)
        if logger.handlers:
            return logger

        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        log_dir = Path(os.getenv("HOLOCINE_LOG_DIR", ".")).expanduser().resolve()
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            # Fallback to current working directory if target path is not writable
            log_dir = Path.cwd()
            log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "blockswap.log"

        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler(stream=sys.stdout)
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        logger.debug("Initialized block swap logger at %s", log_path)
        return logger

    def __init__(self, device="cuda", torch_dtype=torch.bfloat16, tokenizer_path=None):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16, time_division_factor=4, time_division_remainder=1
        )
        # ``BasePipeline`` definitions differ across packages; some initialize ``cpu_offload``
        # while others rely solely on VRAM management flags.  Ensure the attribute exists so
        # later logic that toggles CPU offload remains compatible regardless of the base class
        # version we inherit from.
        if not hasattr(self, "cpu_offload"):
            self.cpu_offload = False
        self.logger = self._configure_blockswap_logger()
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.prompter = WanPrompter(tokenizer_path=tokenizer_path)
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.dit: WanModel = None
        self.dit2: WanModel = None
        self.vae: WanVideoVAE = None
        self.motion_controller: WanMotionControllerModel = None
        self.vace: VaceWanModel = None
        self.model_names = [
            "text_encoder",
            "image_encoder",
            "dit",
            "dit2",
            "vae",
            "motion_controller",
            "vace",
        ]
        self.in_iteration_models = ("dit", "motion_controller", "vace")
        self.in_iteration_models_2 = ("dit2", "motion_controller", "vace")
        self.unit_runner = PipelineUnitRunner()
        self.units = [
            WanVideoUnit_ShapeChecker(),
            WanVideoUnit_NoiseInitializer(),
            WanVideoUnit_InputVideoEmbedder(),
            WanVideoUnit_PromptEmbedder(),
            WanVideoUnit_ImageEmbedderVAE(),
            WanVideoUnit_ImageEmbedderCLIP(),
            WanVideoUnit_ImageEmbedderFused(),
            WanVideoUnit_FunControl(),
            WanVideoUnit_FunReference(),
            WanVideoUnit_FunCameraControl(),
            WanVideoUnit_SpeedControl(),
            WanVideoUnit_VACE(),
            WanVideoUnit_UnifiedSequenceParallel(),
            WanVideoUnit_TeaCache(),
            WanVideoUnit_CfgMerger(),
            WanVideoUnit_ShotEmbedder(),
    
        ]
        self.model_fn = model_fn_wan_video
        self._auto_memory_plan: Optional[BlockSwapPlan] = None
        self._auto_vram_management_applied: bool = False
        
    
    def load_lora(self, module, path, alpha=1):
        loader = GeneralLoRALoader(torch_dtype=self.torch_dtype, device=self.device)
        lora = load_state_dict(path, torch_dtype=self.torch_dtype, device=self.device)
        loader.load(module, lora, alpha=alpha)


    def _log_gpu_memory_state(self, context: str, level: int = logging.DEBUG) -> None:
        level_name = {
            logging.CRITICAL: "CRITICAL",
            logging.ERROR: "ERROR",
            logging.WARNING: "WARNING",
            logging.INFO: "INFO",
            logging.DEBUG: "DEBUG",
        }.get(level, "INFO")
        snapshot = self._get_gpu_memory_snapshot()
        if snapshot is None:
            message = f"{context} GPU memory snapshot unavailable"
            self.logger.log(level, message)
            return
        self.logger.log(
            level,
            (
                f"{context} GPU memory: free≈{snapshot.free_gb:.2f}GB, "
                f"allocated≈{snapshot.allocated_gb:.2f}GB, total≈{snapshot.total_gb:.2f}GB"
            ),
        )


    def _estimate_block_swap_window(
        self,
        latents: torch.Tensor,
        conditioning: Optional[torch.Tensor],
        limit_gb: float,
        target_dtype: Optional[torch.dtype] = None,
    ) -> int:
        if latents is None:
            return 0
        element_size = torch.empty((), dtype=target_dtype or latents.dtype).element_size()
        # Estimate per-frame cost for latents
        per_frame_elements = latents[:, :, :1].numel()
        per_frame_bytes = per_frame_elements * element_size
        if conditioning is not None and conditioning.ndim >= 3 and conditioning.shape[2] == latents.shape[2]:
            conditioning_dtype = target_dtype or conditioning.dtype
            per_frame_bytes += conditioning[:, :, :1].numel() * torch.empty((), dtype=conditioning_dtype).element_size()
        if per_frame_bytes == 0:
            return latents.shape[2]
        safety = 0.85
        available_bytes = max(int(limit_gb * (1024 ** 3) * safety), per_frame_bytes)
        window = max(1, available_bytes // per_frame_bytes)
        return window

    def _get_gpu_memory_snapshot(
        self, device: Optional[Union[str, torch.device]] = None
    ) -> Optional[GPUMemorySnapshot]:
        if device is None:
            device = torch.device(self.device)
        elif not isinstance(device, torch.device):
            device = torch.device(device)
        if device.type != "cuda" or not torch.cuda.is_available():
            return None
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        except RuntimeError:
            return None
        allocated_bytes = torch.cuda.memory_allocated(device)
        return GPUMemorySnapshot(total_bytes=total_bytes, free_bytes=free_bytes, allocated_bytes=allocated_bytes)

    @staticmethod
    def _estimate_module_bytes(module: Optional[torch.nn.Module]) -> int:
        if module is None:
            return 0
        total = 0
        for param in module.parameters():
            total += param.numel() * param.element_size()
        for buffer in module.buffers():
            total += buffer.numel() * buffer.element_size()
        return total

    def _estimate_iteration_model_bytes(self) -> int:
        model_names = set(self.in_iteration_models) | set(getattr(self, "in_iteration_models_2", ()))
        total = 0
        for name in model_names:
            total += self._estimate_module_bytes(getattr(self, name, None))
        return total

    def _plan_block_swap_strategy(
        self,
        latents: torch.Tensor,
        conditioning: Optional[torch.Tensor],
        limit_gb: float,
        window_size: Optional[int],
        window_stride: Optional[int],
        offload_device: torch.device,
        target_dtype: torch.dtype,
        prefer_model_offload: bool,
        force_block_swap: bool,
    ) -> BlockSwapPlan:
        total_frames = latents.shape[2]
        element_size = torch.empty((), dtype=target_dtype).element_size()
        per_frame_elements = latents[:, :, :1].numel()
        per_frame_bytes = per_frame_elements * element_size
        if conditioning is not None and conditioning.ndim >= 3 and conditioning.shape[2] == total_frames:
            cond_size = torch.empty((), dtype=target_dtype).element_size()
            per_frame_bytes += conditioning[:, :, :1].numel() * cond_size

        total_latent_bytes = per_frame_bytes * total_frames
        total_latent_gb = total_latent_bytes / (1024 ** 3)

        memory_snapshot = self._get_gpu_memory_snapshot()
        safety = 0.85
        free_bytes = memory_snapshot.free_bytes if memory_snapshot is not None else None
        available_bytes = int(free_bytes * safety) if free_bytes is not None else None

        limit_bytes = int(limit_gb * (1024 ** 3)) if limit_gb is not None else None

        model_bytes = self._estimate_iteration_model_bytes()
        model_gb = model_bytes / (1024 ** 3)

        prefer_model_offload_applied = False
        if (
            prefer_model_offload
            and memory_snapshot is not None
            and (available_bytes is None or total_latent_bytes > available_bytes)
        ):
            potential_free_bytes = memory_snapshot.free_bytes + model_bytes
            potential_available_bytes = int(potential_free_bytes * safety)
            limit_allows = limit_bytes is None or total_latent_bytes <= limit_bytes
            if limit_allows and total_latent_bytes <= potential_available_bytes:
                available_bytes = potential_available_bytes
                prefer_model_offload_applied = True

        available_gb = (
            available_bytes / (1024 ** 3)
            if available_bytes is not None
            else None
        )

        limit_candidates = []
        if limit_bytes is not None:
            limit_candidates.append(limit_bytes)
        if available_bytes is not None:
            limit_candidates.append(available_bytes)
        if len(limit_candidates) == 1:
            effective_limit_bytes = max(per_frame_bytes, limit_candidates[0])
        elif len(limit_candidates) > 1:
            effective_limit_bytes = max(per_frame_bytes, min(limit_candidates))
        else:
            effective_limit_bytes = max(per_frame_bytes, total_latent_bytes, 1024 ** 3)
        effective_limit_gb = effective_limit_bytes / (1024 ** 3)

        auto_window = max(1, min(total_frames, effective_limit_bytes // per_frame_bytes))
        if window_size is not None:
            sliding_window_size = max(1, min(total_frames, window_size))
        else:
            sliding_window_size = auto_window
        if window_stride is not None:
            sliding_window_stride = max(1, min(sliding_window_size, window_stride))
        else:
            sliding_window_stride = max(1, sliding_window_size // 2)

        if force_block_swap and total_frames > 1 and sliding_window_size >= total_frames:
            fallback_window = max(1, min(total_frames - 1, max(1, total_frames // 2)))
            sliding_window_size = min(sliding_window_size, fallback_window)
            sliding_window_stride = max(1, min(sliding_window_size, sliding_window_stride))

        use_block_swap = force_block_swap or sliding_window_size < total_frames
        if total_frames <= 1:
            use_block_swap = False
        reason = "latents fit after offloading models" if prefer_model_offload_applied else ""
        if force_block_swap and total_frames > 1:
            reason = (reason + "; " if reason else "") + "block swap requested"
        runtime_multiplier = 1.35
        safety_runtime = 0.9

        if not use_block_swap:
            estimated_peak_bytes = int(total_latent_bytes * runtime_multiplier)
            if memory_snapshot is not None:
                peak_available_bytes = memory_snapshot.free_bytes
                if prefer_model_offload_applied:
                    peak_available_bytes += model_bytes
                peak_budget_bytes = int(peak_available_bytes * safety_runtime)
            else:
                peak_budget_bytes = None
            if peak_budget_bytes is not None and estimated_peak_bytes > peak_budget_bytes:
                use_block_swap = True
                reason = "peak usage exceeds available VRAM"
                if window_size is None:
                    sliding_window_size = auto_window
                    sliding_window_stride = max(1, sliding_window_size // 2)
            else:
                if reason == "":
                    reason = "full latent fits in VRAM"

        window_latent_bytes = per_frame_bytes * (sliding_window_size if use_block_swap else total_frames)
        window_latent_gb = window_latent_bytes / (1024 ** 3)
        self.logger.info(
            (
                f"[BlockSwap] Window estimation: total_frames={total_frames}, "
                f"per_frame≈{per_frame_bytes / (1024 ** 2):.3f}MB, "
                f"window_size={sliding_window_size}, stride={sliding_window_stride}, "
                f"window_bytes≈{window_latent_bytes / (1024 ** 3):.3f}GB"
            )
        )

        self.logger.debug(
            (
                f"[BlockSwap] Model footprint estimate: bytes={model_bytes}, "
                f"≈{model_gb:.3f}GB across in-iteration models"
            )
        )

        offload_models = prefer_model_offload_applied
        vram_limit_gb = None
        if memory_snapshot is not None:
            budget_bytes = int(memory_snapshot.free_bytes * safety)
            if prefer_model_offload_applied and available_bytes is not None:
                budget_bytes = available_bytes
            requirement_bytes = window_latent_bytes + model_bytes
            if requirement_bytes > budget_bytes:
                offload_models = True
                budget_gb = budget_bytes / (1024 ** 3)
                vram_limit_gb = max(1.0, budget_gb - window_latent_gb)
                if reason == "":
                    reason = "models exceed VRAM budget"
        else:
            if use_block_swap:
                reason = reason or "cpu execution"
        if memory_snapshot is not None:
            self.logger.debug(
                (
                    "[BlockSwap] GPU snapshot: free≈%.3fGB, allocated≈%.3fGB, total≈%.3fGB"
                    % (memory_snapshot.free_gb, memory_snapshot.allocated_gb, memory_snapshot.total_gb)
                )
            )
        available_display = f"{available_gb:.3f}" if available_gb is not None else "n/a"
        self.logger.debug(
            (
                f"[BlockSwap] Effective limits: limit_gb={effective_limit_gb:.3f}, "
                f"available_gb={available_display}, "
                f"runtime_multiplier={runtime_multiplier}, safety={safety_runtime}"
            )
        )

        storage_device = offload_device if use_block_swap else torch.device(self.device)
        storage_dtype = target_dtype if use_block_swap else (self.torch_dtype or latents.dtype)

        config = None
        if use_block_swap and sliding_window_size < total_frames:
            config = BlockSwapConfig(
                offload_device=storage_device,
                offload_dtype=storage_dtype,
                sliding_window_size=sliding_window_size,
                sliding_window_stride=sliding_window_stride,
                limit_gb=effective_limit_gb,
            )

        return BlockSwapPlan(
            use_block_swap=use_block_swap,
            config=config,
            storage_device=storage_device,
            storage_dtype=storage_dtype,
            available_gb=available_gb,
            total_latent_gb=total_latent_gb,
            window_latent_gb=window_latent_gb,
            model_gb=model_gb,
            window_size=sliding_window_size,
            window_stride=sliding_window_stride,
            effective_limit_gb=effective_limit_gb,
            offload_models=offload_models,
            vram_limit_gb=vram_limit_gb,
            reason=reason,
        )

    def _configure_block_swap(
        self,
        inputs_shared: dict,
        limit_gb: float,
        window_size: Optional[int],
        window_stride: Optional[int],
        offload_device: str,
        offload_dtype: Optional[torch.dtype],
        prefer_model_offload: bool,
        force_block_swap: bool,
    ) -> Optional[BlockSwapConfig]:
        latents: Optional[torch.Tensor] = inputs_shared.get("latents")
        if latents is None:
            warnings.warn("Block swap requested but no latents tensor is available; ignoring request.")
            return None

        self._log_gpu_memory_state("[BlockSwap] Pre-configuration")

        plan = self._plan_block_swap_strategy(
            latents=latents,
            conditioning=inputs_shared.get("y"),
            limit_gb=limit_gb,
            window_size=window_size,
            window_stride=window_stride,
            offload_device=torch.device(offload_device),
            target_dtype=offload_dtype or latents.dtype,
            prefer_model_offload=prefer_model_offload,
            force_block_swap=force_block_swap,
        )

        self.logger.info(
            (
                f"[BlockSwap] Plan decision: use_block_swap={plan.use_block_swap}, "
                f"reason='{plan.reason or 'n/a'}', window={plan.window_size}, stride={plan.window_stride}, "
                f"latents_total≈{plan.total_latent_gb:.3f}GB, window≈{plan.window_latent_gb:.3f}GB, "
                f"models≈{plan.model_gb:.3f}GB"
            )
        )
        if plan.offload_models:
            limit_desc = (
                f"{plan.vram_limit_gb:.3f}GB"
                if plan.vram_limit_gb is not None
                else "unbounded"
            )
            self.logger.info(
                f"[BlockSwap] Model offload required. VRAM limit applied to modules≈{limit_desc}"
            )
        elif not plan.use_block_swap:
            self.logger.info("[BlockSwap] Block swapping not required; processing entire latent batch in-memory.")

        storage_device = plan.storage_device
        storage_dtype = plan.storage_dtype

        latents = latents.to(device=storage_device, dtype=storage_dtype)
        inputs_shared["latents"] = latents
        self.logger.info(
            (
                f"[BlockSwap] Latents moved to {storage_device} ({storage_dtype}) with "
                f"shape={tuple(latents.shape)}, bytes≈{latents.numel() * latents.element_size() / (1024 ** 3):.3f}GB"
            )
        )

        heavy_tensor_keys = [
            "y",
            "reference_latents",
            "vace_context",
            "control_camera_latents_input",
            "first_frame_latents",
        ]
        if plan.use_block_swap:
            for key in heavy_tensor_keys:
                tensor = inputs_shared.get(key)
                if tensor is not None and isinstance(tensor, torch.Tensor):
                    inputs_shared[key] = tensor.to(device=storage_device, dtype=storage_dtype)
                    moved = inputs_shared[key]
                    bytes_gb = moved.numel() * moved.element_size() / (1024 ** 3)
                    self.logger.info(
                        (
                            f"[BlockSwap] {key} moved to {storage_device} ({storage_dtype}) "
                            f"shape={tuple(moved.shape)}, bytes≈{bytes_gb:.3f}GB"
                        )
                    )

        self.cpu_offload = self.cpu_offload or plan.offload_models
        if plan.offload_models and not self.vram_management_enabled and not self._auto_vram_management_applied:
            self.enable_vram_management(vram_limit=plan.vram_limit_gb)
            self._auto_vram_management_applied = True
            limit_display = f"{plan.vram_limit_gb:.2f}GB" if plan.vram_limit_gb is not None else "unbounded"
            self.logger.info(f"[BlockSwap] Enabled VRAM management with limit={limit_display}")

        self._auto_memory_plan = plan

        summary_parts = [
            f"[BlockSwap] Strategy={'enabled' if plan.use_block_swap else 'disabled'}",
            f"window={plan.window_size}",
            f"stride={plan.window_stride}",
            f"latents_window≈{plan.window_latent_gb:.2f}GB",
            f"latents_total≈{plan.total_latent_gb:.2f}GB",
            f"model_mem≈{plan.model_gb:.2f}GB",
        ]
        if plan.available_gb is not None:
            summary_parts.append(f"gpu_free≈{plan.available_gb:.2f}GB")
        summary_parts.append(f"offload_models={'yes' if plan.offload_models else 'no'}")
        if plan.reason:
            summary_parts.append(f"reason={plan.reason}")
        self.logger.info(", ".join(summary_parts))
        self._log_gpu_memory_state("[BlockSwap] Post-configuration")

        return plan.config if plan.use_block_swap else None

        
    def training_loss(self, **inputs):
        max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * self.scheduler.num_train_timesteps)
        min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * self.scheduler.num_train_timesteps)
        timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
        timestep = self.scheduler.timesteps[timestep_id].to(dtype=self.torch_dtype, device=self.device)
        
        inputs["latents"] = self.scheduler.add_noise(inputs["input_latents"], inputs["noise"], timestep)
        training_target = self.scheduler.training_target(inputs["input_latents"], inputs["noise"], timestep)

        
        
        noise_pred = self.model_fn(
            **inputs,
            timestep=timestep,
            logger=self.logger,
            auto_memory_plan=getattr(self, "_auto_memory_plan", None),
            gpu_memory_snapshot_fn=self._get_gpu_memory_snapshot,
        )
        
        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        loss = loss * self.scheduler.training_weight(timestep)
        return loss


    def enable_vram_management(self, num_persistent_param_in_dit=None, vram_limit=None, vram_buffer=0.5):
        self.vram_management_enabled = True
        float8_dtypes = {
            getattr(torch, name)
            for name in (
                "float8_e4m3fn",
                "float8_e4m3fnuz",
                "float8_e5m2",
                "float8_e5m2fnuz",
            )
            if hasattr(torch, name)
        }

        def resolve_computation_dtype(param_dtype: torch.dtype) -> torch.dtype:
            target_dtype = self.torch_dtype if param_dtype in float8_dtypes and self.torch_dtype is not None else param_dtype
            return target_dtype

        if num_persistent_param_in_dit is not None:
            vram_limit = None
        else:
            if vram_limit is None:
                vram_limit = self.get_vram()
            vram_limit = vram_limit - vram_buffer
        if self.text_encoder is not None:
            dtype = next(iter(self.text_encoder.parameters())).dtype
            computation_dtype = resolve_computation_dtype(dtype)
            enable_vram_management(
                self.text_encoder,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Embedding: AutoWrappedModule,
                    T5RelativeEmbedding: AutoWrappedModule,
                    T5LayerNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=computation_dtype,
                    onload_device="cpu",
                    computation_dtype=computation_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.dit is not None:
            dtype = next(iter(self.dit.parameters())).dtype
            device = "cpu" if vram_limit is not None else self.device
            computation_dtype = resolve_computation_dtype(dtype)
            enable_vram_management(
                self.dit,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Embedding: AutoWrappedModule,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: WanAutoCastLayerNorm,
                    RMSNorm: AutoWrappedModule,
                    torch.nn.Conv2d: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=computation_dtype,
                    onload_device=device,
                    computation_dtype=computation_dtype,
                    computation_device=self.device,
                ),
                max_num_param=num_persistent_param_in_dit,
                overflow_module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=computation_dtype,
                    onload_device="cpu",
                    computation_dtype=computation_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.dit2 is not None:
            dtype = next(iter(self.dit2.parameters())).dtype
            device = "cpu" if vram_limit is not None else self.device
            computation_dtype = resolve_computation_dtype(dtype)
            enable_vram_management(
                self.dit2,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: WanAutoCastLayerNorm,
                    RMSNorm: AutoWrappedModule,
                    torch.nn.Conv2d: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=computation_dtype,
                    onload_device=device,
                    computation_dtype=computation_dtype,
                    computation_device=self.device,
                ),
                max_num_param=num_persistent_param_in_dit,
                overflow_module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=computation_dtype,
                    onload_device="cpu",
                    computation_dtype=computation_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.vae is not None:
            dtype = next(iter(self.vae.parameters())).dtype
            computation_dtype = resolve_computation_dtype(dtype)
            enable_vram_management(
                self.vae,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    RMS_norm: AutoWrappedModule,
                    CausalConv3d: AutoWrappedModule,
                    Upsample: AutoWrappedModule,
                    torch.nn.SiLU: AutoWrappedModule,
                    torch.nn.Dropout: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=computation_dtype,
                    onload_device=self.device,
                    computation_dtype=computation_dtype,
                    computation_device=self.device,
                ),
            )
        if self.image_encoder is not None:
            dtype = next(iter(self.image_encoder.parameters())).dtype
            computation_dtype = resolve_computation_dtype(dtype)
            enable_vram_management(
                self.image_encoder,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=computation_dtype,
                    onload_device="cpu",
                    computation_dtype=computation_dtype,
                    computation_device=self.device,
                ),
            )
        if self.motion_controller is not None:
            dtype = next(iter(self.motion_controller.parameters())).dtype
            computation_dtype = resolve_computation_dtype(dtype)
            enable_vram_management(
                self.motion_controller,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=computation_dtype,
                    onload_device="cpu",
                    computation_dtype=computation_dtype,
                    computation_device=self.device,
                ),
            )
        if self.vace is not None:
            dtype = next(iter(self.vace.parameters())).dtype
            device = "cpu" if vram_limit is not None else self.device
            computation_dtype = resolve_computation_dtype(dtype)
            enable_vram_management(
                self.vace,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                    RMSNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=computation_dtype,
                    onload_device=device,
                    computation_dtype=computation_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
            
            
    def initialize_usp(self):
        import torch.distributed as dist
        from xfuser.core.distributed import initialize_model_parallel, init_distributed_environment
        dist.init_process_group(backend="nccl", init_method="env://")
        init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())
        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=1,
            ulysses_degree=dist.get_world_size(),
        )
        torch.cuda.set_device(dist.get_rank())
            
            
    def enable_usp(self):
        from xfuser.core.distributed import get_sequence_parallel_world_size
        from ..distributed.xdit_context_parallel import usp_attn_forward, usp_dit_forward

        for block in self.dit.blocks:
            block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
        self.dit.forward = types.MethodType(usp_dit_forward, self.dit)
        if self.dit2 is not None:
            for block in self.dit2.blocks:
                block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
            self.dit2.forward = types.MethodType(usp_dit_forward, self.dit2)
        self.sp_size = get_sequence_parallel_world_size()
        self.use_unified_sequence_parallel = True


    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/*"),
        redirect_common_files: bool = True,
        use_usp=False,
        custom_config: dict = {},
    ):
        # Redirect model path
        if redirect_common_files:
            redirect_dict = {
                "models_t5_umt5-xxl-enc-bf16.pth": "Wan-AI/Wan2.1-T2V-1.3B",
                "Wan2.1_VAE.pth": "Wan-AI/Wan2.1-T2V-1.3B",
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth": "Wan-AI/Wan2.1-I2V-14B-480P",
            }
            for model_config in model_configs:
                if model_config.origin_file_pattern is None or model_config.model_id is None:
                    continue
                if model_config.origin_file_pattern in redirect_dict and model_config.model_id != redirect_dict[model_config.origin_file_pattern]:
                    print(f"To avoid repeatedly downloading model files, ({model_config.model_id}, {model_config.origin_file_pattern}) is redirected to ({redirect_dict[model_config.origin_file_pattern]}, {model_config.origin_file_pattern}). You can use `redirect_common_files=False` to disable file redirection.")
                    model_config.model_id = redirect_dict[model_config.origin_file_pattern]
        
        # Initialize pipeline
        pipe = WanVideoHoloCinePipeline(device=device, torch_dtype=torch_dtype)
        if use_usp: pipe.initialize_usp()
        
        # Download and load models
        model_manager = ModelManager(torch_dtype=torch_dtype, device=device)
        for model_config in model_configs:
            model_config.download_if_necessary(use_usp=use_usp)
            model_manager.load_model(
                model_config.path,
                device=model_config.offload_device or device,
                torch_dtype=model_config.offload_dtype or torch_dtype
            )
        
        # Load models
        pipe.text_encoder = model_manager.fetch_model("wan_video_text_encoder")
        dit = model_manager.fetch_model("wan_video_dit", index=2)
        if isinstance(dit, list):
            pipe.dit, pipe.dit2 = dit
        else:
            pipe.dit = dit
        pipe.vae = model_manager.fetch_model("wan_video_vae")
        pipe.image_encoder = model_manager.fetch_model("wan_video_image_encoder")
        pipe.motion_controller = model_manager.fetch_model("wan_video_motion_controller")
        pipe.vace = model_manager.fetch_model("wan_video_vace")
        
        # Size division factor
        if pipe.vae is not None:
            pipe.height_division_factor = pipe.vae.upsampling_factor * 2
            pipe.width_division_factor = pipe.vae.upsampling_factor * 2

        # Initialize tokenizer
        tokenizer_config.download_if_necessary(use_usp=use_usp)
        pipe.prompter.fetch_models(pipe.text_encoder)
        pipe.prompter.fetch_tokenizer(tokenizer_config.path)
        
        # Unified Sequence Parallel
        if use_usp: pipe.enable_usp()
        return pipe


    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str,
        negative_prompt: Optional[str] = "",
        # Image-to-video
        input_image: Optional[Image.Image] = None,
        # First-last-frame-to-video
        end_image: Optional[Image.Image] = None,
        # Video-to-video
        input_video: Optional[list[Image.Image]] = None,
        denoising_strength: Optional[float] = 1.0,
        # ControlNet
        control_video: Optional[list[Image.Image]] = None,
        reference_image: Optional[Image.Image] = None,
        # Camera control
        camera_control_direction: Optional[Literal["Left", "Right", "Up", "Down", "LeftUp", "LeftDown", "RightUp", "RightDown"]] = None,
        camera_control_speed: Optional[float] = 1/54,
        camera_control_origin: Optional[tuple] = (0, 0.532139961, 0.946026558, 0.5, 0.5, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0),
        # VACE
        vace_video: Optional[list[Image.Image]] = None,
        vace_video_mask: Optional[Image.Image] = None,
        vace_reference_image: Optional[Image.Image] = None,
        vace_scale: Optional[float] = 1.0,
        # Randomness
        seed: Optional[int] = None,
        rand_device: Optional[str] = "cpu",
        # Shape
        height: Optional[int] = 480,
        width: Optional[int] = 832,
        num_frames=81,
        # Classifier-free guidance
        cfg_scale: Optional[float] = 5.0,
        cfg_merge: Optional[bool] = False,
        # Boundary
        switch_DiT_boundary: Optional[float] = 0.875,
        # Scheduler
        num_inference_steps: Optional[int] = 50,
        sigma_shift: Optional[float] = 5.0,
        # Speed control
        motion_bucket_id: Optional[int] = None,
        # VAE tiling
        tiled: Optional[bool] = True,
        tile_size: Optional[tuple[int, int]] = (30, 52),
        tile_stride: Optional[tuple[int, int]] = (15, 26),
        # Sliding window
        sliding_window_size: Optional[int] = None,
        sliding_window_stride: Optional[int] = None,
        # Teacache
        tea_cache_l1_thresh: Optional[float] = None,
        tea_cache_model_id: Optional[str] = "",
        # progress_bar
        progress_bar_cmd=tqdm,
        shot_cut_frames: Optional[list[int]] = None,
        shot_mask_type: Optional[Literal["id", "normalized", "alternating"]] = None,
        text_cut_positions: Optional[torch.Tensor] = None,
        # Block swapping
        enable_block_swap: bool = False,
        block_swap_limit_gb: float = 32.0,
        block_swap_size: Optional[int] = None,
        block_swap_stride: Optional[int] = None,
        block_swap_device: str = "cpu",
        block_swap_dtype: Optional[torch.dtype] = None,
        block_swap_prefer_model_offload: bool = True,
    ):
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)
        
        # Inputs
        inputs_posi = {
            "prompt": prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh, "tea_cache_model_id": tea_cache_model_id, "num_inference_steps": num_inference_steps,
        }
        inputs_nega = {
            "negative_prompt": negative_prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh, "tea_cache_model_id": tea_cache_model_id, "num_inference_steps": num_inference_steps,
        }
        inputs_shared = {
            "input_image": input_image,
            "end_image": end_image,
            "input_video": input_video, "denoising_strength": denoising_strength,
            "control_video": control_video, "reference_image": reference_image,
            "camera_control_direction": camera_control_direction, "camera_control_speed": camera_control_speed, "camera_control_origin": camera_control_origin,
            "vace_video": vace_video, "vace_video_mask": vace_video_mask, "vace_reference_image": vace_reference_image, "vace_scale": vace_scale,
            "seed": seed, "rand_device": rand_device,
            "height": height, "width": width, "num_frames": num_frames,
            "cfg_scale": cfg_scale, "cfg_merge": cfg_merge,
            "sigma_shift": sigma_shift,
            "motion_bucket_id": motion_bucket_id,
            "tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride,
            "sliding_window_size": sliding_window_size, "sliding_window_stride": sliding_window_stride,
            "shot_cut_frames": shot_cut_frames,
            "shot_mask_type": shot_mask_type
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        block_swap_config = None
        if enable_block_swap or block_swap_size is not None or block_swap_stride is not None:
            block_swap_config = self._configure_block_swap(
                inputs_shared,
                limit_gb=block_swap_limit_gb,
                window_size=block_swap_size,
                window_stride=block_swap_stride,
                offload_device=block_swap_device,
                offload_dtype=block_swap_dtype,
                prefer_model_offload=block_swap_prefer_model_offload,
                force_block_swap=enable_block_swap,
            )
            if block_swap_config is not None:
                if sliding_window_size is None:
                    sliding_window_size = block_swap_config.sliding_window_size
                if sliding_window_stride is None:
                    sliding_window_stride = block_swap_config.sliding_window_stride
                inputs_shared["computation_device"] = self.device
                inputs_shared["computation_dtype"] = self.torch_dtype
                inputs_shared["sliding_window_size"] = sliding_window_size
                inputs_shared["sliding_window_stride"] = sliding_window_stride
                for tensor_dict in (inputs_posi, inputs_nega):
                    tensor = tensor_dict.get("y")
                    if tensor is not None and isinstance(tensor, torch.Tensor):
                        tensor_dict["y"] = tensor.to(
                            device=block_swap_config.offload_device,
                            dtype=block_swap_config.offload_dtype,
                        )



        # not use mask for negative
        inputs_nega["text_cut_positions"]['global']=None
        inputs_nega["text_cut_positions"]['shots']=[]
        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        num_progress_steps = len(self.scheduler.timesteps)
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            # Switch DiT if necessary
            if timestep.item() < switch_DiT_boundary * self.scheduler.num_train_timesteps and self.dit2 is not None and not models["dit"] is self.dit2:
                self.load_models_to_device(self.in_iteration_models_2)
                models["dit"] = self.dit2

            # Timestep
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)

            # Inference
            noise_pred_posi = self.model_fn(
                **models,
                **inputs_shared,
                **inputs_posi,
                timestep=timestep,
                logger=self.logger,
                auto_memory_plan=getattr(self, "_auto_memory_plan", None),
                gpu_memory_snapshot_fn=self._get_gpu_memory_snapshot,
            )
            if cfg_scale != 1.0:
                if cfg_merge:
                    noise_pred_posi, noise_pred_nega = noise_pred_posi.chunk(2, dim=0)
                else:
                    noise_pred_nega = self.model_fn(
                        **models,
                        **inputs_shared,
                        **inputs_nega,
                        timestep=timestep,
                        logger=self.logger,
                        auto_memory_plan=getattr(self, "_auto_memory_plan", None),
                        gpu_memory_snapshot_fn=self._get_gpu_memory_snapshot,
                    )
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            # Scheduler
            inputs_shared["latents"] = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], inputs_shared["latents"])
            if "first_frame_latents" in inputs_shared:
                inputs_shared["latents"][:, :, 0:1] = inputs_shared["first_frame_latents"]
            if block_swap_config is not None:
                latents_tensor = inputs_shared["latents"].to(
                    device=block_swap_config.offload_device,
                    dtype=block_swap_config.offload_dtype,
                )
                inputs_shared["latents"] = latents_tensor
                latent_bytes = latents_tensor.numel() * latents_tensor.element_size()
                self.logger.info(
                    (
                        f"[BlockSwap] Iteration {progress_id + 1}/{num_progress_steps} "
                        f"offloaded latents to {block_swap_config.offload_device} "
                        f"({block_swap_config.offload_dtype}) size≈{latent_bytes / (1024 ** 3):.2f}GB"
                    )
                )
                self._log_gpu_memory_state(
                    f"[BlockSwap] After iteration {progress_id + 1}/{num_progress_steps}",
                    level=logging.INFO,
                )

        # VACE (TODO: remove it)
        if vace_reference_image is not None:
            inputs_shared["latents"] = inputs_shared["latents"][:, :, 1:]

        # Decode
        self.load_models_to_device(['vae'])
        video = self.vae.decode(inputs_shared["latents"], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        video = self.vae_output_to_video(video)
        self.load_models_to_device([])

        return video



class WanVideoUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width", "num_frames"))

    def process(self, pipe: WanVideoHoloCinePipeline, height, width, num_frames):
        height, width, num_frames = pipe.check_resize_height_width(height, width, num_frames)
        return {"height": height, "width": width, "num_frames": num_frames}



class WanVideoUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width", "num_frames", "seed", "rand_device", "vace_reference_image"))

    def process(self, pipe: WanVideoHoloCinePipeline, height, width, num_frames, seed, rand_device, vace_reference_image):
        length = (num_frames - 1) // 4 + 1
        if vace_reference_image is not None:
            length += 1
        shape = (1, pipe.vae.model.z_dim, length, height // pipe.vae.upsampling_factor, width // pipe.vae.upsampling_factor)
        noise = pipe.generate_noise(shape, seed=seed, rand_device=rand_device)
        if vace_reference_image is not None:
            noise = torch.concat((noise[:, :, -1:], noise[:, :, :-1]), dim=2)
        return {"noise": noise}
    


class WanVideoUnit_InputVideoEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_video", "noise", "tiled", "tile_size", "tile_stride", "vace_reference_image"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoHoloCinePipeline, input_video, noise, tiled, tile_size, tile_stride, vace_reference_image):
        if input_video is None:
            return {"latents": noise}
        pipe.load_models_to_device(["vae"])
        input_video = pipe.preprocess_video(input_video)
        input_latents = pipe.vae.encode(input_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)




        if vace_reference_image is not None:
            vace_reference_image = pipe.preprocess_video([vace_reference_image])
            vace_reference_latents = pipe.vae.encode(vace_reference_image, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
            input_latents = torch.concat([vace_reference_latents, input_latents], dim=2)
        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents}
        else:
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
            return {"latents": latents}



class WanVideoUnit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt", "positive": "positive"},
            input_params_nega={"prompt": "negative_prompt", "positive": "positive"},
            onload_model_names=("text_encoder",)
        )

    def process(self, pipe: WanVideoHoloCinePipeline, prompt, positive) -> dict:
        pipe.load_models_to_device(self.onload_model_names)
        prompt_emb,positions = pipe.prompter.encode_prompt(prompt, positive=positive, device=pipe.device)
        return {"context": prompt_emb,"text_cut_positions":positions}




class WanVideoUnit_Prompt_separatelyEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt", "positive": "positive"},
            input_params_nega={"prompt": "negative_prompt", "positive": "positive"},
            onload_model_names=("text_encoder",)
        )

    def process(self, pipe: WanVideoHoloCinePipeline, prompt, positive) -> dict:
        pipe.load_models_to_device(self.onload_model_names)
        prompt_emb,positions = pipe.prompter.encode_prompt_separately(prompt, positive=positive, device=pipe.device)
        return {"context": prompt_emb,"text_cut_positions":positions}



class WanVideoUnit_ImageEmbedder(PipelineUnit):
    """
    Deprecated
    """
    def __init__(self):
        super().__init__(
            input_params=("input_image", "end_image", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("image_encoder", "vae")
        )

    def process(self, pipe: WanVideoHoloCinePipeline, input_image, end_image, num_frames, height, width, tiled, tile_size, tile_stride):
        if input_image is None or pipe.image_encoder is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
        clip_context = pipe.image_encoder.encode_image([image])
        msk = torch.ones(1, num_frames, height//8, width//8, device=pipe.device)
        msk[:, 1:] = 0
        if end_image is not None:
            end_image = pipe.preprocess_image(end_image.resize((width, height))).to(pipe.device)
            vae_input = torch.concat([image.transpose(0,1), torch.zeros(3, num_frames-2, height, width).to(image.device), end_image.transpose(0,1)],dim=1)
            if pipe.dit.has_image_pos_emb:
                clip_context = torch.concat([clip_context, pipe.image_encoder.encode_image([end_image])], dim=1)
            msk[:, -1:] = 1
        else:
            vae_input = torch.concat([image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image.device)], dim=1)

        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
        msk = msk.transpose(1, 2)[0]
        
        y = pipe.vae.encode([vae_input.to(dtype=pipe.torch_dtype, device=pipe.device)], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        clip_context = clip_context.to(dtype=pipe.torch_dtype, device=pipe.device)
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"clip_feature": clip_context, "y": y}



class WanVideoUnit_ImageEmbedderCLIP(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "end_image", "height", "width"),
            onload_model_names=("image_encoder",)
        )

    def process(self, pipe: WanVideoHoloCinePipeline, input_image, end_image, height, width):
        if input_image is None or pipe.image_encoder is None or not pipe.dit.require_clip_embedding:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
        clip_context = pipe.image_encoder.encode_image([image])
        if end_image is not None:
            end_image = pipe.preprocess_image(end_image.resize((width, height))).to(pipe.device)
            if pipe.dit.has_image_pos_emb:
                clip_context = torch.concat([clip_context, pipe.image_encoder.encode_image([end_image])], dim=1)
        clip_context = clip_context.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"clip_feature": clip_context}
    


class WanVideoUnit_ImageEmbedderVAE(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "end_image", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoHoloCinePipeline, input_image, end_image, num_frames, height, width, tiled, tile_size, tile_stride):
        if input_image is None or not pipe.dit.require_vae_embedding:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
        msk = torch.ones(1, num_frames, height//8, width//8, device=pipe.device)
        msk[:, 1:] = 0
        if end_image is not None:
            end_image = pipe.preprocess_image(end_image.resize((width, height))).to(pipe.device)
            vae_input = torch.concat([image.transpose(0,1), torch.zeros(3, num_frames-2, height, width).to(image.device), end_image.transpose(0,1)],dim=1)
            msk[:, -1:] = 1
        else:
            vae_input = torch.concat([image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image.device)], dim=1)

        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
        msk = msk.transpose(1, 2)[0]
        
        y = pipe.vae.encode([vae_input.to(dtype=pipe.torch_dtype, device=pipe.device)], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"y": y}



class WanVideoUnit_ImageEmbedderFused(PipelineUnit):
    """
    Encode input image to latents using VAE. This unit is for Wan-AI/Wan2.2-TI2V-5B.
    """
    def __init__(self):
        super().__init__(
            input_params=("input_image", "latents", "height", "width", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoHoloCinePipeline, input_image, latents, height, width, tiled, tile_size, tile_stride):
        if input_image is None or not pipe.dit.fuse_vae_embedding_in_latents:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).transpose(0, 1)
        z = pipe.vae.encode([image], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        latents[:, :, 0: 1] = z
        return {"latents": latents, "fuse_vae_embedding_in_latents": True, "first_frame_latents": z}



class WanVideoUnit_FunControl(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("control_video", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride", "clip_feature", "y"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoHoloCinePipeline, control_video, num_frames, height, width, tiled, tile_size, tile_stride, clip_feature, y):
        if control_video is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        control_video = pipe.preprocess_video(control_video)
        control_latents = pipe.vae.encode(control_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        control_latents = control_latents.to(dtype=pipe.torch_dtype, device=pipe.device)
        if clip_feature is None or y is None:
            clip_feature = torch.zeros((1, 257, 1280), dtype=pipe.torch_dtype, device=pipe.device)
            y = torch.zeros((1, 16, (num_frames - 1) // 4 + 1, height//8, width//8), dtype=pipe.torch_dtype, device=pipe.device)
        else:
            y = y[:, -16:]
        y = torch.concat([control_latents, y], dim=1)
        return {"clip_feature": clip_feature, "y": y}
    


class WanVideoUnit_FunReference(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("reference_image", "height", "width", "reference_image"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoHoloCinePipeline, reference_image, height, width):
        if reference_image is None:
            return {}
        pipe.load_models_to_device(["vae"])
        reference_image = reference_image.resize((width, height))
        reference_latents = pipe.preprocess_video([reference_image])
        reference_latents = pipe.vae.encode(reference_latents, device=pipe.device)
        clip_feature = pipe.preprocess_image(reference_image)
        clip_feature = pipe.image_encoder.encode_image([clip_feature])
        return {"reference_latents": reference_latents, "clip_feature": clip_feature}



class WanVideoUnit_FunCameraControl(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames", "camera_control_direction", "camera_control_speed", "camera_control_origin", "latents", "input_image"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoHoloCinePipeline, height, width, num_frames, camera_control_direction, camera_control_speed, camera_control_origin, latents, input_image):
        if camera_control_direction is None:
            return {}
        camera_control_plucker_embedding = pipe.dit.control_adapter.process_camera_coordinates(
            camera_control_direction, num_frames, height, width, camera_control_speed, camera_control_origin)
        
        control_camera_video = camera_control_plucker_embedding[:num_frames].permute([3, 0, 1, 2]).unsqueeze(0)
        control_camera_latents = torch.concat(
            [
                torch.repeat_interleave(control_camera_video[:, :, 0:1], repeats=4, dim=2),
                control_camera_video[:, :, 1:]
            ], dim=2
        ).transpose(1, 2)
        b, f, c, h, w = control_camera_latents.shape
        control_camera_latents = control_camera_latents.contiguous().view(b, f // 4, 4, c, h, w).transpose(2, 3)
        control_camera_latents = control_camera_latents.contiguous().view(b, f // 4, c * 4, h, w).transpose(1, 2)
        control_camera_latents_input = control_camera_latents.to(device=pipe.device, dtype=pipe.torch_dtype)

        input_image = input_image.resize((width, height))
        input_latents = pipe.preprocess_video([input_image])
        pipe.load_models_to_device(self.onload_model_names)
        input_latents = pipe.vae.encode(input_latents, device=pipe.device)
        y = torch.zeros_like(latents).to(pipe.device)
        y[:, :, :1] = input_latents
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"control_camera_latents_input": control_camera_latents_input, "y": y}



class WanVideoUnit_SpeedControl(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("motion_bucket_id",))

    def process(self, pipe: WanVideoHoloCinePipeline, motion_bucket_id):
        if motion_bucket_id is None:
            return {}
        motion_bucket_id = torch.Tensor((motion_bucket_id,)).to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"motion_bucket_id": motion_bucket_id}



class WanVideoUnit_VACE(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("vace_video", "vace_video_mask", "vace_reference_image", "vace_scale", "height", "width", "num_frames", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )

    def process(
        self,
        pipe: WanVideoHoloCinePipeline,
        vace_video, vace_video_mask, vace_reference_image, vace_scale,
        height, width, num_frames,
        tiled, tile_size, tile_stride
    ):
        if vace_video is not None or vace_video_mask is not None or vace_reference_image is not None:
            pipe.load_models_to_device(["vae"])
            if vace_video is None:
                vace_video = torch.zeros((1, 3, num_frames, height, width), dtype=pipe.torch_dtype, device=pipe.device)
            else:
                vace_video = pipe.preprocess_video(vace_video)
            
            if vace_video_mask is None:
                vace_video_mask = torch.ones_like(vace_video)
            else:
                vace_video_mask = pipe.preprocess_video(vace_video_mask, min_value=0, max_value=1)
            
            inactive = vace_video * (1 - vace_video_mask) + 0 * vace_video_mask
            reactive = vace_video * vace_video_mask + 0 * (1 - vace_video_mask)
            inactive = pipe.vae.encode(inactive, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
            reactive = pipe.vae.encode(reactive, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
            vace_video_latents = torch.concat((inactive, reactive), dim=1)
            
            vace_mask_latents = rearrange(vace_video_mask[0,0], "T (H P) (W Q) -> 1 (P Q) T H W", P=8, Q=8)
            vace_mask_latents = torch.nn.functional.interpolate(vace_mask_latents, size=((vace_mask_latents.shape[2] + 3) // 4, vace_mask_latents.shape[3], vace_mask_latents.shape[4]), mode='nearest-exact')
            
            if vace_reference_image is None:
                pass
            else:
                vace_reference_image = pipe.preprocess_video([vace_reference_image])
                vace_reference_latents = pipe.vae.encode(vace_reference_image, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
                vace_reference_latents = torch.concat((vace_reference_latents, torch.zeros_like(vace_reference_latents)), dim=1)
                vace_video_latents = torch.concat((vace_reference_latents, vace_video_latents), dim=2)
                vace_mask_latents = torch.concat((torch.zeros_like(vace_mask_latents[:, :, :1]), vace_mask_latents), dim=2)
            
            vace_context = torch.concat((vace_video_latents, vace_mask_latents), dim=1)
            return {"vace_context": vace_context, "vace_scale": vace_scale}
        else:
            return {"vace_context": None, "vace_scale": vace_scale}



class WanVideoUnit_UnifiedSequenceParallel(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=())

    def process(self, pipe: WanVideoHoloCinePipeline):
        if hasattr(pipe, "use_unified_sequence_parallel"):
            if pipe.use_unified_sequence_parallel:
                return {"use_unified_sequence_parallel": True}
        return {}



class WanVideoUnit_TeaCache(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"num_inference_steps": "num_inference_steps", "tea_cache_l1_thresh": "tea_cache_l1_thresh", "tea_cache_model_id": "tea_cache_model_id"},
            input_params_nega={"num_inference_steps": "num_inference_steps", "tea_cache_l1_thresh": "tea_cache_l1_thresh", "tea_cache_model_id": "tea_cache_model_id"},
        )

    def process(self, pipe: WanVideoHoloCinePipeline, num_inference_steps, tea_cache_l1_thresh, tea_cache_model_id):
        if tea_cache_l1_thresh is None:
            return {}
        return {"tea_cache": TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh, model_id=tea_cache_model_id)}



class WanVideoUnit_CfgMerger(PipelineUnit):
    def __init__(self):
        super().__init__(take_over=True)
        self.concat_tensor_names = ["context", "clip_feature", "y", "reference_latents", "shot_indices"]

    def process(self, pipe: WanVideoHoloCinePipeline, inputs_shared, inputs_posi, inputs_nega):
        if not inputs_shared["cfg_merge"]:
            return inputs_shared, inputs_posi, inputs_nega
        for name in self.concat_tensor_names:
            tensor_posi = inputs_posi.get(name)
            tensor_nega = inputs_nega.get(name)
            tensor_shared = inputs_shared.get(name)
            if tensor_posi is not None and tensor_nega is not None:
                inputs_shared[name] = torch.concat((tensor_posi, tensor_nega), dim=0)
            elif tensor_shared is not None:
                inputs_shared[name] = torch.concat((tensor_shared, tensor_shared), dim=0)
        inputs_posi.clear()
        inputs_nega.clear()
        return inputs_shared, inputs_posi, inputs_nega



class WanVideoUnit_ShotEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("shot_cut_frames", "num_frames"))

    def process(self, pipe: WanVideoHoloCinePipeline, shot_cut_frames, num_frames):
        if shot_cut_frames is None:
            return {}
        
        num_latent_frames = (num_frames - 1) // 4 + 1
        
        # Convert frame cut indices to latent cut indices
        shot_cut_latents = [0]
        for frame_idx in sorted(shot_cut_frames):
            if frame_idx > 0:
                latent_idx = (frame_idx - 1) // 4 + 1
                if latent_idx < num_latent_frames:
                    shot_cut_latents.append(latent_idx)
        
        cuts = sorted(list(set(shot_cut_latents))) + [num_latent_frames]


        shot_indices = torch.zeros(num_latent_frames, dtype=torch.long)
        for i in range(len(cuts) - 1):
            start_latent, end_latent = cuts[i], cuts[i+1]
            shot_indices[start_latent:end_latent] = i
            
        shot_indices = shot_indices.unsqueeze(0).to(device=pipe.device)
        
        return {"shot_indices": shot_indices}


class TeaCache:
    def __init__(self, num_inference_steps, rel_l1_thresh, model_id):
        self.num_inference_steps = num_inference_steps
        self.step = 0
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        self.rel_l1_thresh = rel_l1_thresh
        self.previous_residual = None
        self.previous_hidden_states = None
        
        self.coefficients_dict = {
            "Wan2.1-T2V-1.3B": [-5.21862437e+04, 9.23041404e+03, -5.28275948e+02, 1.36987616e+01, -4.99875664e-02],
            "Wan2.1-T2V-14B": [-3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01],
            "Wan2.1-I2V-14B-480P": [2.57151496e+05, -3.54229917e+04,  1.40286849e+03, -1.35890334e+01, 1.32517977e-01],
            "Wan2.1-I2V-14B-720P": [ 8.10705460e+03,  2.13393892e+03, -3.72934672e+02,  1.66203073e+01, -4.17769401e-02],
        }
        if model_id not in self.coefficients_dict:
            supported_model_ids = ", ".join([i for i in self.coefficients_dict])
            raise ValueError(f"{model_id} is not a supported TeaCache model id. Please choose a valid model id in ({supported_model_ids}).")
        self.coefficients = self.coefficients_dict[model_id]

    def check(self, dit: WanModel, x, t_mod):
        modulated_inp = t_mod.clone()
        if self.step == 0 or self.step == self.num_inference_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            coefficients = self.coefficients
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance += rescale_func(((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = modulated_inp
        self.step += 1
        if self.step == self.num_inference_steps:
            self.step = 0
        if should_calc:
            self.previous_hidden_states = x.clone()
        return not should_calc

    def store(self, hidden_states):
        self.previous_residual = hidden_states - self.previous_hidden_states
        self.previous_hidden_states = None

    def update(self, hidden_states):
        hidden_states = hidden_states + self.previous_residual
        return hidden_states



class TemporalTiler_BCTHW:
    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        auto_memory_plan: Optional[BlockSwapPlan] = None,
        gpu_memory_snapshot_fn: Optional[
            Callable[[Optional[Union[str, torch.device]]], Optional[GPUMemorySnapshot]]
        ] = None,
    ):
        self.logger = logger or logging.getLogger(BLOCK_SWAP_LOGGER_NAME)
        self._auto_memory_plan = auto_memory_plan
        self._gpu_memory_snapshot_fn = gpu_memory_snapshot_fn

    def _log(self, level: int, message: str) -> None:
        if self.logger is not None and self.logger.handlers:
            self.logger.log(level, message)
        else:
            print(message, flush=True)

    def _info(self, message: str) -> None:
        self._log(logging.INFO, message)

    def _warning(self, message: str) -> None:
        self._log(logging.WARNING, message)

    def _get_gpu_memory_snapshot(
        self, device: Optional[Union[str, torch.device]] = None
    ) -> Optional[GPUMemorySnapshot]:
        if self._gpu_memory_snapshot_fn is None:
            return None
        try:
            if device is not None:
                return self._gpu_memory_snapshot_fn(device)
            return self._gpu_memory_snapshot_fn()
        except TypeError:
            # Fallback for callables that ignore the device parameter
            return self._gpu_memory_snapshot_fn()

    @staticmethod
    def _estimate_total_windows(total_frames: int, window_size: int, stride: int) -> int:
        return max(1, math.ceil(max(0, total_frames - window_size) / max(1, stride)) + 1)

    @staticmethod
    def _is_oom_error(error: RuntimeError) -> bool:
        message = str(error).lower()
        return "out of memory" in message or "cuda error" in message

    def _adjust_window_to_available_vram(
        self,
        window_total_bytes: int,
        current_window_size: int,
        computation_device,
    ) -> Tuple[int, Optional[str], Optional[GPUMemorySnapshot]]:
        candidate_size = current_window_size
        reason_parts = []
        plan = self._auto_memory_plan
        plan_limit_gb = getattr(plan, "effective_limit_gb", None) if plan is not None else None
        limit_bytes = None
        if plan_limit_gb is not None:
            limit_bytes = int(plan_limit_gb * (1024 ** 3))
        snapshot: Optional[GPUMemorySnapshot] = None
        free_budget_bytes: Optional[int] = None
        if self._gpu_memory_snapshot_fn is not None:
            target_device = computation_device
            if target_device is not None and not isinstance(target_device, torch.device):
                target_device = torch.device(target_device)
            if (
                isinstance(target_device, torch.device)
                and target_device.type == "cuda"
                and torch.cuda.is_available()
            ):
                snapshot = self._gpu_memory_snapshot_fn(target_device)
        if snapshot is not None:
            free_budget_bytes = int(snapshot.free_bytes * 0.9)
            if free_budget_bytes <= 0:
                free_budget_bytes = snapshot.free_bytes
        if limit_bytes is not None and window_total_bytes > limit_bytes:
            target_bytes = max(1, int(limit_bytes * 0.95))
            ratio = target_bytes / max(1, window_total_bytes)
            candidate_size = max(1, int(current_window_size * ratio))
            if candidate_size >= current_window_size and current_window_size > 1:
                candidate_size = current_window_size - 1
            reason_parts.append(f"plan limit {plan_limit_gb:.2f}GB")
        if free_budget_bytes is not None and window_total_bytes > free_budget_bytes:
            ratio = free_budget_bytes / max(1, window_total_bytes)
            reduced = max(1, int(current_window_size * ratio))
            if reduced >= current_window_size and current_window_size > 1:
                reduced = current_window_size - 1
            if reduced < candidate_size:
                candidate_size = reduced
            elif reduced > candidate_size:
                candidate_size = min(candidate_size, reduced)
            reason_parts.append(f"free VRAM≈{snapshot.free_gb:.2f}GB")
        if candidate_size > current_window_size:
            candidate_size = current_window_size
        reason = " & ".join(reason_parts) if reason_parts else None
        if reason is not None and candidate_size == current_window_size and current_window_size > 1:
            candidate_size = current_window_size - 1
        if candidate_size < 1:
            candidate_size = 1
        return candidate_size, reason, snapshot

    def build_1d_mask(self, length, left_bound, right_bound, border_width):
        x = torch.ones((length,))
        if border_width == 0:
            return x

        shift = 0.5
        if not left_bound:
            x[:border_width] = (torch.arange(border_width) + shift) / border_width
        if not right_bound:
            x[-border_width:] = torch.flip((torch.arange(border_width) + shift) / border_width, dims=(0,))
        return x

    def build_mask(self, data, is_bound, border_width):
        _, _, T, _, _ = data.shape
        t = self.build_1d_mask(T, is_bound[0], is_bound[1], border_width[0])
        mask = repeat(t, "T -> 1 1 T 1 1")
        return mask
    
    def run(
        self,
        model_fn,
        sliding_window_size,
        sliding_window_stride,
        computation_device=None,
        computation_dtype=None,
        model_kwargs=None,
        tensor_names=None,
        batch_size=None,
    ):
        if model_kwargs is None:
            model_kwargs = {}
        if tensor_names is None:
            tensor_names = []
        tensor_names = [tensor_name for tensor_name in tensor_names if model_kwargs.get(tensor_name) is not None]
        if len(tensor_names) == 0:
            return model_fn(**model_kwargs)
        tensor_dict = {tensor_name: model_kwargs[tensor_name] for tensor_name in tensor_names}
        static_kwargs = {key: value for key, value in model_kwargs.items() if key not in tensor_names}
        B, C, T, H, W = tensor_dict[tensor_names[0]].shape
        if batch_size is not None:
            B *= batch_size
        data_device, data_dtype = tensor_dict[tensor_names[0]].device, tensor_dict[tensor_names[0]].dtype
        if computation_device is None:
            computation_device = data_device
        if computation_dtype is None:
            computation_dtype = data_dtype

        def slice_time_dimension(tensor: torch.Tensor) -> torch.Tensor:
            if tensor.shape == ():
                return tensor
            target_length = T
            for dim, size in enumerate(tensor.shape):
                if size == target_length:
                    indexer = [slice(None)] * tensor.ndim
                    indexer[dim] = slice(t, t_end)
                    return tensor[tuple(indexer)]
            indexer = [slice(None)] * tensor.ndim
            dim = 2 if tensor.ndim > 2 else tensor.ndim - 1
            indexer[dim] = slice(t, t_end)
            return tensor[tuple(indexer)]
        value = torch.zeros((B, C, T, H, W), device=data_device, dtype=data_dtype)
        weight = torch.zeros((1, 1, T, 1, 1), device=data_device, dtype=data_dtype)
        current_window_size = sliding_window_size
        current_stride = max(1, min(sliding_window_stride, current_window_size))
        total_windows = self._estimate_total_windows(T, current_window_size, current_stride)
        self._info(
            (
                f"[BlockSwap] Sliding window execution: windows={total_windows}, size={current_window_size}, "
                f"stride={current_stride}, tensors={','.join(tensor_names)}"
            )
        )
        window_index = 0
        min_window_size_used = current_window_size
        max_window_size_used = current_window_size
        t = 0
        dtype_element_size = torch.empty((), dtype=computation_dtype).element_size()
        while t < T:
            while True:
                window_size = min(current_window_size, T - t)
                t_end = t + window_size
                tensor_slices = []
                window_total_bytes = 0
                for tensor_name in tensor_names:
                    tensor_slice = slice_time_dimension(tensor_dict[tensor_name])
                    tensor_slices.append((tensor_name, tensor_slice))
                    window_total_bytes += tensor_slice.numel() * dtype_element_size
                adjusted_size, adjustment_reason, snapshot = self._adjust_window_to_available_vram(
                    window_total_bytes,
                    current_window_size,
                    computation_device,
                )
                if (
                    adjustment_reason is not None
                    and adjusted_size < current_window_size
                    and current_window_size > 1
                ):
                    self._info(
                        (
                            f"[BlockSwap] Shrinking window frames[{t}:{t_end}) from {current_window_size} "
                            f"to {adjusted_size} due to {adjustment_reason}"
                        )
                    )
                    current_window_size = adjusted_size
                    current_stride = max(1, min(sliding_window_stride, current_window_size))
                    total_windows = self._estimate_total_windows(T, current_window_size, current_stride)
                    self._info(
                        (
                            f"[BlockSwap] Updated sliding window parameters: windows={total_windows}, "
                            f"size={current_window_size}, stride={current_stride}"
                        )
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                break
            window_size = min(current_window_size, T - t)
            t_end = t + window_size
            updated_tensors = {}
            for tensor_name, tensor_slice in tensor_slices:
                updated_tensors[tensor_name] = tensor_slice.to(
                    device=computation_device, dtype=computation_dtype
                )
            window_total_bytes = sum(
                updated_tensors[name].numel() * updated_tensors[name].element_size()
                for name in tensor_names
            )
            snapshot = None
            if (
                isinstance(computation_device, torch.device)
                and computation_device.type == "cuda"
                and torch.cuda.is_available()
            ):
                snapshot = self._get_gpu_memory_snapshot(computation_device)
            if snapshot is not None:
                self._info(
                    (
                        f"[BlockSwap] Swapping window {window_index + 1}/{total_windows} "
                        f"frames[{t}:{t_end}) size={window_size} (≈{window_total_bytes / (1024 ** 3):.2f}GB) "
                        f"-> {computation_device} ({computation_dtype}) | free≈{snapshot.free_gb:.2f}GB "
                        f"allocated≈{snapshot.allocated_gb:.2f}GB"
                    )
                )
            else:
                self._info(
                    (
                        f"[BlockSwap] Swapping window {window_index + 1}/{total_windows} "
                        f"frames[{t}:{t_end}) size={window_size} (≈{window_total_bytes / (1024 ** 3):.2f}GB) "
                        f"-> {computation_device} ({computation_dtype})"
                    )
                )
            try:
                call_kwargs = dict(static_kwargs)
                call_kwargs.update(updated_tensors)
                model_output = model_fn(**call_kwargs).to(device=data_device, dtype=data_dtype)
            except RuntimeError as error:
                if not self._is_oom_error(error) or current_window_size == 1:
                    raise
                new_window_size = max(1, current_window_size // 2)
                if new_window_size == current_window_size:
                    raise
                self._warning(
                    (
                        f"[BlockSwap] OOM detected for window frames[{t}:{t_end}); "
                        f"reducing window size from {current_window_size} to {new_window_size}"
                    )
                )
                current_window_size = new_window_size
                current_stride = max(1, min(sliding_window_stride, current_window_size))
                total_windows = self._estimate_total_windows(T, current_window_size, current_stride)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self._info(
                    (
                        f"[BlockSwap] Updated sliding window parameters: windows={total_windows}, "
                        f"size={current_window_size}, stride={current_stride}"
                    )
                )
                del updated_tensors
                continue
            border_width = max(0, current_window_size - current_stride)
            mask = self.build_mask(
                model_output,
                is_bound=(t == 0, t_end == T),
                border_width=(border_width,)
            ).to(device=data_device, dtype=data_dtype)
            value[:, :, t:t_end, :, :] += model_output * mask
            weight[:, :, t:t_end, :, :] += mask
            del updated_tensors
            window_index += 1
            min_window_size_used = min(min_window_size_used, window_size)
            max_window_size_used = max(max_window_size_used, window_size)
            t += current_stride
        self._info(
            (
                f"[BlockSwap] Sliding window execution complete: processed {window_index} windows "
                f"(window_size range={min_window_size_used}-{max_window_size_used})"
            )
        )
        value /= weight
        return value



def model_fn_wan_video(
    dit: WanModel,
    motion_controller: WanMotionControllerModel = None,
    vace: VaceWanModel = None,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    reference_latents = None,
    vace_context = None,
    vace_scale = 1.0,
    tea_cache: TeaCache = None,
    use_unified_sequence_parallel: bool = False,
    motion_bucket_id: Optional[torch.Tensor] = None,
    sliding_window_size: Optional[int] = None,
    sliding_window_stride: Optional[int] = None,
    cfg_merge: bool = False,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    control_camera_latents_input = None,
    fuse_vae_embedding_in_latents: bool = False,
    shot_indices: Optional[torch.Tensor] = None,
    shot_mask_type: Optional[str] = None,
    text_cut_positions: Optional[torch.Tensor] = None,
    computation_device: Optional[torch.device] = None,
    computation_dtype: Optional[torch.dtype] = None,
    logger: Optional[logging.Logger] = None,
    auto_memory_plan: Optional[BlockSwapPlan] = None,
    gpu_memory_snapshot_fn: Optional[
        Callable[[Optional[Union[str, torch.device]]], Optional[GPUMemorySnapshot]]
    ] = None,
    **kwargs,
):
    if sliding_window_size is not None and sliding_window_stride is not None:
        model_kwargs = dict(
            dit=dit,
            motion_controller=motion_controller,
            vace=vace,
            latents=latents,
            timestep=timestep,
            context=context,
            clip_feature=clip_feature,
            y=y,
            reference_latents=reference_latents,
            vace_context=vace_context,
            vace_scale=vace_scale,
            tea_cache=tea_cache,
            use_unified_sequence_parallel=use_unified_sequence_parallel,
            motion_bucket_id=motion_bucket_id,
            shot_indices=shot_indices,
            shot_mask_type=shot_mask_type,
            text_cut_positions=text_cut_positions,
            logger=logger,
            auto_memory_plan=auto_memory_plan,
            gpu_memory_snapshot_fn=gpu_memory_snapshot_fn,
        )
        tiler = TemporalTiler_BCTHW(
            logger=logger,
            auto_memory_plan=auto_memory_plan,
            gpu_memory_snapshot_fn=gpu_memory_snapshot_fn,
        )
        return tiler.run(
            model_fn_wan_video,
            sliding_window_size,
            sliding_window_stride,
            computation_device=computation_device or timestep.device,
            computation_dtype=computation_dtype or latents.dtype,
            model_kwargs=model_kwargs,
            tensor_names=["latents", "y"],
            batch_size=2 if cfg_merge else 1
        )
    
    if use_unified_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import (get_sequence_parallel_rank,
                                            get_sequence_parallel_world_size,
                                            get_sp_group)

    x = latents
    if shot_mask_type is not None and shot_indices is not None:
        num_shots = shot_indices.max() + 1
        if shot_mask_type == "id":
            shot_mask_tensor = shot_indices.to(x.dtype)
        elif shot_mask_type == "normalized":
            shot_mask_tensor = shot_indices.to(x.dtype) / (20) if num_shots > 1 else torch.zeros_like(shot_indices, dtype=x.dtype)
        elif shot_mask_type == "alternating":
            shot_mask_tensor = (shot_indices % 2).to(x.dtype)
        else:
            shot_mask_tensor = None
        if shot_mask_tensor is not None:
            b, c, f, h, w = x.shape
            mask = shot_mask_tensor.view(b, 1, f, 1, 1).expand(b, 1, f, h, w)
            x = torch.cat([x, mask], dim=1)


    # Timestep
    if dit.seperated_timestep and fuse_vae_embedding_in_latents:
        timestep = torch.concat([
            torch.zeros((1, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device),
            torch.ones((latents.shape[2] - 1, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device) * timestep
        ]).flatten()
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).unsqueeze(0))
        t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))
    else:
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    
    # Motion Controller
    if motion_bucket_id is not None and motion_controller is not None:
        t_mod = t_mod + motion_controller(motion_bucket_id).unflatten(1, (6, dit.dim))
    context = dit.text_embedding(context)

    # Merged cfg
    if x.shape[0] != context.shape[0]:
        x = torch.concat([x] * context.shape[0], dim=0)
    if timestep.shape[0] != context.shape[0]:
        timestep = torch.concat([timestep] * context.shape[0], dim=0)

    # Image Embedding
    if y is not None and dit.require_vae_embedding:
        x = torch.cat([x, y], dim=1)
    if clip_feature is not None and dit.require_clip_embedding:
        clip_embdding = dit.img_emb(clip_feature)
        context = torch.cat([clip_embdding, context], dim=1)
    
    # Add camera control
    x, (f, h, w) = dit.patchify(x, control_camera_latents_input)
    
    if dit.shot_embedding is not None and shot_indices is not None:
        assert shot_indices.shape[0] == x.shape[0], f"Batch size mismatch between latents ({x.shape[0]}) and shot_indices ({shot_indices.shape[0]})"
        assert shot_indices.shape[-1] == f, f"Shot indices length mismatch. Expected {f}, got {shot_indices.shape[-1]}"
        shot_ids = shot_indices.repeat_interleave(h * w, dim=1)
        shot_embs = dit.shot_embedding(shot_ids)
        x = x + shot_embs
    
    # Reference image
    if reference_latents is not None:
        if len(reference_latents.shape) == 5:
            reference_latents = reference_latents[:, :, 0]
        reference_latents = dit.ref_conv(reference_latents).flatten(2).transpose(1, 2)
        x = torch.concat([reference_latents, x], dim=1)
        f += 1
    
    freqs = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
        


   
    # TeaCache
    if tea_cache is not None:
        tea_cache_update = tea_cache.check(dit, x, t_mod)
    else:
        tea_cache_update = False
        
    if vace_context is not None:
        vace_hints = vace(x, vace_context, context, t_mod, freqs)


    use_attn_mask = True
    if use_attn_mask:

            if text_cut_positions['global']==None:
                attn_mask=None
            else:
                
                try:


                    B, S_q = x.shape[0], x.shape[1]


                    num_img_tokens = clip_embdding.shape[1] if (clip_feature is not None and dit.require_clip_embedding) else 0
                    L_text_ctx = context.shape[1] - num_img_tokens

                    g0, g1 = map(int, text_cut_positions['global'])
                    shot_ranges = text_cut_positions['shots']
                    S_shots = len(shot_ranges)

                    #debug
                    max_shot_id = shot_indices.max()
                    num_defined_shots = S_shots
                    assert max_shot_id < num_defined_shots, \
                        f"Error: Shot index out of bounds! The maximum shot ID in the data is {max_shot_id.item()}, " \
                        f"but only {num_defined_shots} shots were defined (valid IDs are from 0 to {num_defined_shots - 1}). " \
                        f"Please check your `shot_indices` and `text_cut_positions` inputs! " \
                        f"prompt: {kwargs['prompt']}"


                    max_end = max([g1] + [int(r[1]) for r in shot_ranges])
                    L_text_pos = max_end + 1


                    device, dtype = x.device, x.dtype

        
                    global_mask = torch.zeros(L_text_ctx, dtype=torch.bool, device=device)
                    global_mask[g0:g1+1] = True

                    if shot_ranges!=[]:


                        shot_table = torch.zeros(S_shots, L_text_ctx, dtype=torch.bool, device=device)
                        for sid, (s0, s1) in enumerate(shot_ranges):
                            s0 = int(s0); s1 = int(s1)
                            shot_table[sid, s0:s1+1] = True

                        frame_shot_indices = shot_indices
                        if frame_shot_indices.dim() > 2:
                            frame_shot_indices = frame_shot_indices.reshape(frame_shot_indices.shape[0], -1)
                        frame_shot_indices = frame_shot_indices.to(device=device, dtype=torch.long)

                        allow_shot = shot_table[frame_shot_indices]

                        if allow_shot.shape[0] != B:
                            allow_shot = allow_shot.expand(B, -1, -1)

                        tokens_per_frame = h * w if h is not None and w is not None else None
                        if tokens_per_frame and allow_shot.shape[1] * tokens_per_frame == S_q:
                            allow_shot = allow_shot.repeat_interleave(tokens_per_frame, dim=1)
                        elif allow_shot.shape[1] != S_q:
                            base = max(allow_shot.shape[1], 1)
                            repeat_factor = math.ceil(S_q / base)
                            allow_shot = allow_shot.repeat_interleave(repeat_factor, dim=1)[..., :S_q, :]

                        allow = allow_shot | global_mask.view(1, 1, L_text_ctx)
                    else:
                        allow = global_mask.view(1, 1, L_text_ctx)

                    pad_mask = torch.zeros(L_text_ctx, dtype=torch.bool, device=device)
                    if L_text_pos < L_text_ctx:
                        pad_mask[L_text_pos:] = True
                    allow = allow | pad_mask.view(1, 1, L_text_ctx)

                    block_value = -1e4
                    bias = torch.zeros(B, S_q, L_text_ctx, dtype=dtype, device=device)
                    bias = bias.masked_fill(~allow, block_value)

       
                    attn_mask = bias.unsqueeze(1)

                except Exception as e:
                    print(f"!!!!!! ERROR FOUND !!!!!!!")
                    print(f"Error message: {e}")


       
                    attn_mask=None
    else:
        attn_mask = None
    


    use_sparse_self_attn = getattr(dit, 'use_sparse_self_attn', False)
    if use_sparse_self_attn and shot_indices is not None:
        shot_indices_for_cuts = shot_indices
        if shot_indices_for_cuts.dim() == 1:
            shot_indices_for_cuts = shot_indices_for_cuts.unsqueeze(0)
        if reference_latents is not None:
            ref_token_count = reference_latents.shape[1]
            tokens_per_frame = h * w
            if tokens_per_frame > 0 and ref_token_count > 0:
                ref_frames, remainder = divmod(ref_token_count, tokens_per_frame)
                if remainder != 0:
                    if logger is not None:
                        logger.warning(
                            "Reference latents token count (%s) is not a multiple of tokens per frame (%s); "
                            "ignoring reference tokens for sparse attention grouping.",
                            ref_token_count,
                            tokens_per_frame,
                        )
                elif ref_frames > 0:
                    max_shot_ids = shot_indices_for_cuts.amax(dim=1, keepdim=True)
                    ref_offsets = torch.arange(
                        ref_frames,
                        device=shot_indices_for_cuts.device,
                        dtype=shot_indices_for_cuts.dtype,
                    ).unsqueeze(0)
                    ref_labels = max_shot_ids + 1 + ref_offsets
                    shot_indices_for_cuts = torch.cat([ref_labels, shot_indices_for_cuts], dim=1)
        shot_latent_indices = shot_indices_for_cuts.repeat_interleave(h * w, dim=1)
        shot_latent_indices = labels_to_cuts(shot_latent_indices)
    else:
        shot_latent_indices = None
    


    # blocks
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            x = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
    if tea_cache_update:
        x = tea_cache.update(x)
    else:
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward
        
        for block_id, block in enumerate(dit.blocks):
            
            if use_gradient_checkpointing_offload:
                with torch.autograd.graph.save_on_cpu():
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, t_mod, freqs,attn_mask,shot_latent_indices,h*w,
                        use_reentrant=False,
                    )
            elif use_gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, context, t_mod, freqs,attn_mask,shot_latent_indices,h*w,
                    use_reentrant=False,
                )
            else:
                x = block(x, context, t_mod, freqs, attn_mask,shot_latent_indices,h*w)
            if vace_context is not None and block_id in vace.vace_layers_mapping:
                current_vace_hint = vace_hints[vace.vace_layers_mapping[block_id]]
                if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
                    current_vace_hint = torch.chunk(current_vace_hint, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
                x = x + current_vace_hint * vace_scale
        if tea_cache is not None:
            tea_cache.store(x)
            
    x = dit.head(x, t)
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            x = get_sp_group().all_gather(x, dim=1)
    # Remove reference latents
    if reference_latents is not None:
        x = x[:, reference_latents.shape[1]:]
        f -= 1
    x = dit.unpatchify(x, (f, h, w))
    return x



def labels_to_cuts(batch_labels: torch.Tensor):

    assert batch_labels.dim() == 2, "expect [b, s]"
    b, s = batch_labels.shape
    labs = batch_labels.to(torch.long)


    diffs = torch.zeros((b, s), dtype=torch.bool, device=labs.device)
    diffs[:, 1:] = labs[:, 1:] != labs[:, :-1]

    cuts_list = []
    for i in range(b):

        change_pos = torch.nonzero(diffs[i], as_tuple=False).flatten()  
        cuts = [0]
        cuts.extend(change_pos.tolist())
        if cuts[-1] != s:
            cuts.append(s)

        cuts_list.append(cuts)
    return cuts_list
