import sys

import numpy as np
import torch

from diffsynth.models.utils import load_state_dict
from diffsynth.pipelines.wan_video_holocine import WanVideoHoloCinePipeline
from HoloCine_inference_full_attention import build_model_configs


def test_load_state_dict_from_gguf(monkeypatch, tmp_path):
    class FakeTensor:
        def __init__(self, name):
            self.name = name
            self._data = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        @property
        def data(self):
            return self._data

    class FakeGGUFReader:
        def __init__(self, path):
            self.tensors = [FakeTensor("model.weight")]

    monkeypatch.setitem(sys.modules, "gguf", type("M", (), {"GGUFReader": FakeGGUFReader}))

    file_path = tmp_path / "dummy.gguf"
    file_path.write_bytes(b"fake")

    state_dict = load_state_dict(str(file_path))
    assert "model.weight" in state_dict
    tensor = state_dict["model.weight"]
    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype in {torch.float32, torch.float16, torch.bfloat16}
    assert torch.allclose(tensor.to(torch.float32), torch.tensor([1.0, 2.0, 3.0]))


def test_apply_lightning_lora_targets_all_modules(tmp_path):
    pipe = WanVideoHoloCinePipeline(device="cpu")
    pipe.dit = torch.nn.Linear(2, 2)
    pipe.dit2 = torch.nn.Linear(2, 2)

    calls = []

    def fake_load_lora(module, path, alpha=1.0):
        calls.append((module, path, alpha))

    pipe.load_lora = fake_load_lora  # type: ignore[assignment]

    lora_path = tmp_path / "mock.safetensors"
    lora_path.write_text("mock")

    pipe.apply_lightning_lora([str(lora_path)], alpha=0.5)
    expected = [
        (pipe.dit, str(lora_path), 0.5),
        (pipe.dit2, str(lora_path), 0.5),
    ]
    assert calls == expected


def test_build_model_configs_quant_suffix(tmp_path):
    configs = build_model_configs(
        str(tmp_path),
        use_quantized=True,
        quant_suffix="Q4_K_M",
    )
    assert any(path.endswith("Q4_K_M.gguf") for path in (cfg.path for cfg in configs))
