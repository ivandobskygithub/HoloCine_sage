import pathlib
import sys

import torch

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from diffsynth.pipelines.wan_video_holocine import WanVideoHoloCinePipeline
from diffsynth.vram_management.layers import AutoWrappedLinear


def test_pipeline_uses_distinct_computation_dtype_for_float8_storage():
    pipe = WanVideoHoloCinePipeline(device="cpu", torch_dtype=torch.float8_e4m3fn)
    assert pipe.torch_dtype == torch.float8_e4m3fn
    assert pipe.computation_dtype is not None
    assert pipe.computation_dtype != torch.float8_e4m3fn


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
