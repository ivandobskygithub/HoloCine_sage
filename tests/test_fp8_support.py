import pathlib
import sys

import pytest
import torch

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from diffsynth.pipelines.wan_video_holocine import (
    TemporalTiler_BCTHW,
    WanVideoHoloCinePipeline,
)
from diffsynth.schedulers.flow_match import FlowMatchScheduler
from diffsynth.vram_management.layers import AutoWrappedLinear


def test_pipeline_uses_distinct_computation_dtype_for_float8_storage():
    pipe = WanVideoHoloCinePipeline(device="cpu", torch_dtype=torch.float8_e4m3fn)
    assert pipe.torch_dtype == torch.float8_e4m3fn
    assert pipe.computation_dtype is not None
    assert pipe.computation_dtype != torch.float8_e4m3fn


def _preferred_float8_dtype():
    for name in ("float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2", "float8_e5m2fnuz"):
        if hasattr(torch, name):
            return getattr(torch, name)
    return None


def test_pipeline_tracks_latent_storage_dtype_defaults():
    pipe = WanVideoHoloCinePipeline(device="cpu", torch_dtype=torch.float16)
    preferred_fp8 = _preferred_float8_dtype()
    if preferred_fp8 is not None:
        assert pipe.latent_storage_dtype == preferred_fp8
    else:
        assert pipe.latent_storage_dtype == torch.float16

    pipe.to(dtype=torch.float32)
    if preferred_fp8 is not None:
        assert pipe.latent_storage_dtype == preferred_fp8
    else:
        assert pipe.latent_storage_dtype == torch.float32

    pipe.set_latent_storage_dtype(torch.float16)
    assert pipe.latent_storage_dtype == torch.float16

    pipe.set_latent_storage_dtype(None)
    if preferred_fp8 is not None:
        assert pipe.latent_storage_dtype == preferred_fp8
    else:
        assert pipe.latent_storage_dtype == pipe.torch_dtype


def test_pipeline_respects_explicit_computation_dtype():
    pipe = WanVideoHoloCinePipeline(
        device="cpu",
        torch_dtype=torch.float8_e4m3fn,
        computation_dtype=torch.float16,
    )
    assert pipe.computation_dtype == torch.float16
    pipe.to(dtype=torch.float8_e4m3fn)
    assert pipe.computation_dtype == torch.float16
    pipe.set_computation_dtype(None)
    assert pipe.computation_dtype != torch.float8_e4m3fn


def test_auto_wrapped_linear_casts_fp8_weights_for_linear():
    base = torch.nn.Linear(4, 3, bias=False).to(dtype=torch.float8_e4m3fn)
    wrapper = AutoWrappedLinear(
        base,
        offload_dtype=torch.float8_e4m3fn,
        offload_device="cpu",
        onload_dtype=torch.float16,
        onload_device="cpu",
        computation_dtype=torch.float16,
        computation_device="cpu",
        vram_limit=None,
    )

    x = torch.randn(2, 4, dtype=torch.float16)
    out = wrapper(x)
    assert out.dtype == torch.float16

    x_fp8 = torch.randn(2, 4).to(dtype=torch.float8_e4m3fn)
    out_fp8 = wrapper(x_fp8)
    assert out_fp8.dtype == torch.float16


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="float8 not supported")
def test_pipeline_casts_latents_to_storage_dtype():
    pipe = WanVideoHoloCinePipeline(
        device="cpu",
        torch_dtype=torch.bfloat16,
        latent_storage_dtype=torch.float8_e4m3fn,
    )

    latents = torch.zeros((1, 2, 4, 3, 3), dtype=torch.bfloat16)
    first_frame = latents[:, :, :1].clone()
    inputs = {"latents": latents.clone(), "first_frame_latents": first_frame.clone()}

    pipe._ensure_latent_storage(inputs)

    assert inputs["latents"].dtype == torch.float8_e4m3fn
    assert inputs["first_frame_latents"].dtype == torch.float8_e4m3fn


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="float8 not supported")
def test_temporal_tiler_accumulates_in_runtime_dtype_before_downcasting():
    latents = torch.zeros((1, 2, 4, 3, 3), dtype=torch.float8_e4m3fn)

    tiler = TemporalTiler_BCTHW()

    def model_fn(latents):
        assert latents.dtype == torch.float16
        return torch.ones_like(latents, dtype=torch.float16)

    output = tiler.run(
        model_fn,
        sliding_window_size=2,
        sliding_window_stride=2,
        computation_device="cpu",
        computation_dtype=torch.float16,
        model_kwargs={"latents": latents},
        tensor_names=["latents"],
        return_to_storage=True,
    )

    assert output.dtype == torch.float8_e4m3fn
    assert output.device == latents.device
    assert torch.allclose(output.to(dtype=torch.float16), torch.ones_like(latents, dtype=torch.float16))


def test_pipeline_can_disable_default_fp8_storage(monkeypatch):
    monkeypatch.setenv("HOLOCINE_DISABLE_FP8_STORAGE", "1")

    pipe = WanVideoHoloCinePipeline(device="cpu", torch_dtype=torch.float16)

    assert pipe.latent_storage_dtype == torch.float16

    pipe.to(dtype=torch.float32)
    assert pipe.latent_storage_dtype == torch.float32

    pipe.set_latent_storage_dtype(None)
    assert pipe.latent_storage_dtype == pipe.torch_dtype


def test_pipeline_can_opt_in_to_fp8_runtime(monkeypatch):
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("float8 not supported")

    monkeypatch.setenv("HOLOCINE_ENABLE_FP8_COMPUTE", "1")

    def fake_support(self):
        return True

    monkeypatch.setattr(WanVideoHoloCinePipeline, "_is_fp8_compute_supported", fake_support)

    pipe = WanVideoHoloCinePipeline(device="cpu", torch_dtype=torch.float8_e4m3fn)
    assert pipe.computation_dtype == torch.float8_e4m3fn

    monkeypatch.delenv("HOLOCINE_ENABLE_FP8_COMPUTE")


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="float8 not supported")
def test_flow_match_scheduler_handles_fp8_latents_without_promotion_errors():
    scheduler = FlowMatchScheduler(num_inference_steps=4)
    scheduler.set_timesteps(4)

    sample = torch.zeros((2, 3), dtype=torch.float8_e4m3fn)
    model_output = torch.ones((2, 3), dtype=torch.bfloat16)
    timestep = scheduler.timesteps[0]

    prev_sample = scheduler.step(model_output, timestep, sample)

    assert prev_sample.dtype == torch.float8_e4m3fn

    sigma = scheduler.sigmas[0]
    sigma_next = scheduler.sigmas[1]
    expected = sample.to(torch.float32) + model_output.to(torch.float32) * (sigma_next - sigma)

    assert torch.allclose(prev_sample.to(torch.float32), expected, atol=1e-2, rtol=1e-2)
