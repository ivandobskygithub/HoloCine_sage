import math
import pathlib
import sys

import torch

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from diffsynth.pipelines.wan_video_holocine import (
    BlockSwapPlan,
    GPUMemorySnapshot,
    WanVideoHoloCinePipeline,
)


def _to_bytes(gb: float) -> int:
    return int(gb * (1024 ** 3))


def make_snapshot(total_gb: float, free_gb: float, allocated_gb: float) -> GPUMemorySnapshot:
    return GPUMemorySnapshot(
        total_bytes=_to_bytes(total_gb),
        free_bytes=_to_bytes(free_gb),
        allocated_bytes=_to_bytes(allocated_gb),
    )


def test_plan_prefers_model_offload_when_latents_fit_after_offload():
    pipe = WanVideoHoloCinePipeline(device="cuda", torch_dtype=torch.float16)
    pipe._get_gpu_memory_snapshot = lambda device=None: make_snapshot(40.0, 0.25, 39.75)
    pipe._estimate_iteration_model_bytes = lambda: _to_bytes(0.1)

    latents = torch.zeros((1, 4, 30, 1000, 1000), dtype=torch.float16)

    plan = pipe._plan_block_swap_strategy(
        latents=latents,
        conditioning=None,
        limit_gb=0.25,
        window_size=None,
        window_stride=None,
        offload_device=torch.device("cpu"),
        target_dtype=torch.float16,
        prefer_model_offload=True,
    )

    assert plan.use_block_swap is False
    assert plan.offload_models is True
    assert math.isclose(plan.window_size, latents.shape[2])
    assert "offloading models" in plan.reason


def test_plan_retains_block_swap_when_latents_exceed_limit():
    pipe = WanVideoHoloCinePipeline(device="cuda", torch_dtype=torch.float16)
    pipe._get_gpu_memory_snapshot = lambda device=None: make_snapshot(40.0, 0.25, 39.75)
    pipe._estimate_iteration_model_bytes = lambda: _to_bytes(0.1)

    latents = torch.zeros((1, 4, 30, 1000, 1000), dtype=torch.float16)

    plan = pipe._plan_block_swap_strategy(
        latents=latents,
        conditioning=None,
        limit_gb=0.05,
        window_size=None,
        window_stride=None,
        offload_device=torch.device("cpu"),
        target_dtype=torch.float16,
        prefer_model_offload=True,
    )

    assert plan.use_block_swap is True



def test_configure_block_swap_keeps_latents_on_pipeline_device_when_not_swapping():
    pipe = WanVideoHoloCinePipeline(device="cpu", torch_dtype=torch.float16)

    latents = torch.randn((1, 2, 4, 8, 8), dtype=torch.float16)
    inputs = {"latents": latents.clone()}

    plan = BlockSwapPlan(
        use_block_swap=False,
        config=None,
        storage_device=torch.device("cpu"),
        storage_dtype=torch.float16,
        available_gb=1.5,
        total_latent_gb=0.01,
        window_latent_gb=0.01,
        model_gb=0.02,
        window_size=latents.shape[2],
        window_stride=latents.shape[2],
        effective_limit_gb=1.5,
        offload_models=True,
        vram_limit_gb=0.5,
        reason="latents fit after offloading models",
    )

    pipe._plan_block_swap_strategy = lambda **_: plan

    result = pipe._configure_block_swap(
        inputs_shared=inputs,
        limit_gb=1.0,
        window_size=None,
        window_stride=None,
        offload_device="cpu",
        offload_dtype=torch.float16,
        prefer_model_offload=True,
    )

    assert result is None
    assert pipe._auto_memory_plan is plan
    assert pipe.cpu_offload is True
    assert inputs["latents"].device.type == "cpu"
    assert inputs["latents"].dtype == torch.float16
