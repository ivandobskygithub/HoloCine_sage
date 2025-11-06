import math
import pathlib
import sys

import torch
import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from diffsynth.pipelines.wan_video_holocine import (
    BlockSwapConfig,
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
        storage_dtype=torch.float16,
        runtime_dtype=torch.float16,
        prefer_model_offload=True,
        force_block_swap=False,
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
        storage_dtype=torch.float16,
        runtime_dtype=torch.float16,
        prefer_model_offload=True,
        force_block_swap=False,
    )

    assert plan.use_block_swap is True





def test_plan_can_offload_and_block_swap_when_requested():
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
        storage_dtype=torch.float16,
        runtime_dtype=torch.float16,
        prefer_model_offload=True,
        force_block_swap=True,
    )

    assert plan.use_block_swap is True
    assert plan.offload_models is True
    assert plan.config is not None
    assert plan.config.sliding_window_size < latents.shape[2]
    assert "block swap requested" in plan.reason


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
        storage_total_gb=0.01,
        storage_window_gb=0.01,
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
        force_block_swap=True,
    )

    assert result is None
    assert pipe._auto_memory_plan is plan
    assert pipe.cpu_offload is True
    assert inputs["latents"].device.type == "cpu"
    assert inputs["latents"].dtype == torch.float16


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="float8 not supported")
def test_configure_block_swap_preserves_float8_storage_when_not_swapping():
    pipe = WanVideoHoloCinePipeline(
        device="cpu",
        torch_dtype=torch.bfloat16,
        latent_storage_dtype=torch.float8_e4m3fn,
    )

    latents = torch.zeros((1, 2, 4, 8, 8), dtype=torch.float8_e4m3fn)
    inputs = {"latents": latents.clone()}

    plan = BlockSwapPlan(
        use_block_swap=False,
        config=None,
        storage_device=torch.device("cpu"),
        storage_dtype=torch.float8_e4m3fn,
        available_gb=1.5,
        total_latent_gb=0.01,
        window_latent_gb=0.01,
        storage_total_gb=0.01,
        storage_window_gb=0.01,
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
        offload_dtype=torch.float8_e4m3fn,
        prefer_model_offload=True,
        force_block_swap=True,
    )

    assert result is None
    assert pipe._auto_memory_plan is plan
    assert inputs["latents"].dtype == torch.float8_e4m3fn


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="float8 not supported")
def test_configure_block_swap_defaults_to_pipeline_latent_dtype():
    pipe = WanVideoHoloCinePipeline(
        device="cpu",
        torch_dtype=torch.bfloat16,
        latent_storage_dtype=torch.float8_e4m3fn,
    )

    latents = torch.zeros((1, 2, 16, 8, 8), dtype=torch.bfloat16)
    inputs = {"latents": latents.clone()}

    captured = {}

    def fake_plan(**kwargs):
        captured.update(kwargs)
        window = max(1, kwargs["latents"].shape[2] // 2)
        config = BlockSwapConfig(
            offload_device=kwargs["offload_device"],
            offload_dtype=kwargs["storage_dtype"],
            sliding_window_size=window,
            sliding_window_stride=window,
            limit_gb=kwargs["limit_gb"],
        )
        return BlockSwapPlan(
            use_block_swap=True,
            config=config,
            storage_device=kwargs["offload_device"],
            storage_dtype=kwargs["storage_dtype"],
            available_gb=None,
            total_latent_gb=0.0,
            window_latent_gb=0.0,
            storage_total_gb=0.0,
            storage_window_gb=0.0,
            model_gb=0.0,
            window_size=window,
            window_stride=window,
            effective_limit_gb=kwargs["limit_gb"],
            offload_models=False,
            vram_limit_gb=None,
            reason="forced",
        )

    pipe._plan_block_swap_strategy = fake_plan

    config = pipe._configure_block_swap(
        inputs_shared=inputs,
        limit_gb=0.0005,
        window_size=None,
        window_stride=None,
        offload_device="cpu",
        offload_dtype=None,
        prefer_model_offload=False,
        force_block_swap=True,
    )

    assert captured["storage_dtype"] == torch.float8_e4m3fn
    assert config is not None
    assert pipe._auto_memory_plan.storage_dtype == torch.float8_e4m3fn
    assert inputs["latents"].dtype == torch.float8_e4m3fn
    assert config.offload_dtype == torch.float8_e4m3fn


def test_plan_reports_runtime_and_storage_latent_sizes_separately():
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("float8 not supported in this PyTorch build")

    pipe = WanVideoHoloCinePipeline(device="cuda", torch_dtype=torch.float8_e4m3fn, computation_dtype=torch.bfloat16)
    pipe._get_gpu_memory_snapshot = lambda device=None: None
    pipe._estimate_iteration_model_bytes = lambda: 0

    latents = torch.zeros((1, 2, 4, 8, 8), dtype=torch.float8_e4m3fn)

    plan = pipe._plan_block_swap_strategy(
        latents=latents,
        conditioning=None,
        limit_gb=10.0,
        window_size=None,
        window_stride=None,
        offload_device=torch.device("cpu"),
        storage_dtype=torch.float8_e4m3fn,
        runtime_dtype=torch.bfloat16,
        prefer_model_offload=False,
        force_block_swap=False,
    )

    per_frame_elements = latents[:, :, :1].numel()
    runtime_bytes = per_frame_elements * torch.empty((), dtype=torch.bfloat16).element_size() * latents.shape[2]
    storage_bytes = per_frame_elements * torch.empty((), dtype=torch.float8_e4m3fn).element_size() * latents.shape[2]

    assert math.isclose(plan.total_latent_gb, runtime_bytes / (1024 ** 3))
    assert math.isclose(plan.storage_total_gb, storage_bytes / (1024 ** 3))
    assert math.isclose(plan.window_latent_gb, plan.total_latent_gb)
    assert math.isclose(plan.storage_window_gb, plan.storage_total_gb)
