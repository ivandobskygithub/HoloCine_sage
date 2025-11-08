import numpy as np
import torch

from diffsynth.models import utils as model_utils
from diffsynth.pipelines import wan_video_holocine
from diffsynth.utils import ModelConfig


class _DummyTensor:
    def __init__(self, name: str, array: np.ndarray):
        self.name = name
        self._array = array

    @property
    def data(self):
        return self._array


class _DummyReader:
    def __init__(self, path: str):
        self.path = path
        self.tensors = [
            _DummyTensor("dummy.weight", np.array([[1.0, -1.0]], dtype=np.float32)),
            _DummyTensor("dummy.bias", np.array([0.5], dtype=np.float32)),
        ]


class _DummyGGUFModule:
    GGUFReader = _DummyReader


def test_load_state_dict_from_gguf(monkeypatch, tmp_path):
    checkpoint_path = tmp_path / "mock.gguf"
    checkpoint_path.write_bytes(b"GGUF")

    monkeypatch.setattr(model_utils, "gguf", _DummyGGUFModule)

    state_dict = model_utils.load_state_dict(str(checkpoint_path), torch_dtype=torch.float32)

    assert set(state_dict.keys()) == {"dummy.weight", "dummy.bias"}
    assert all(tensor.device.type == "cpu" for tensor in state_dict.values())
    assert all(tensor.dtype == torch.float32 for tensor in state_dict.values())


def test_model_config_applies_lora(monkeypatch, tmp_path):
    class _DummyVAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.upsampling_factor = 1

    class FakeModelManager:
        lora_calls = []

        def __init__(self, torch_dtype=None, device=None):
            self.torch_dtype = torch_dtype
            self.device = device
            self.model = []
            self.model_name = []
            self.model_path = []
            self._loaded = {}
            self.lora_calls = []
            self._name_sequence = [
                "wan_video_text_encoder",
                "wan_video_dit",
                "wan_video_dit",
                "wan_video_vae",
            ]

        def _create_module(self, name: str):
            if name == "wan_video_vae":
                return _DummyVAE()
            return torch.nn.Linear(1, 1)

        def load_model(self, path, device=None, torch_dtype=None, model_names=None):
            index = len(self.model)
            name = self._name_sequence[index] if index < len(self._name_sequence) else "wan_video_vace"
            module = self._create_module(name)
            self.model.append(module)
            self.model_name.append(name)
            self.model_path.append(path)
            self._loaded.setdefault(name, []).append(module)

        def load_lora(self, file_path="", state_dict=None, lora_alpha=1.0):
            self.__class__.lora_calls.append((file_path, lora_alpha))

        def fetch_model(self, model_name, index=None, require_model_path=False, file_path=None):
            modules = self._loaded.get(model_name)
            if not modules:
                module = self._create_module(model_name)
                modules = [module]
                self._loaded[model_name] = modules
                self.model.append(module)
                self.model_name.append(model_name)
                self.model_path.append(file_path or "<generated>")
            if index is None:
                result = modules[0]
            elif isinstance(index, int):
                result = modules[:index]
            else:
                result = modules
            if require_model_path:
                return result, file_path or "<generated>"
            return result

    monkeypatch.setattr(wan_video_holocine, "ModelManager", FakeModelManager)
    monkeypatch.setattr(wan_video_holocine.WanPrompter, "fetch_models", lambda self, text_encoder: None)
    monkeypatch.setattr(wan_video_holocine.WanPrompter, "fetch_tokenizer", lambda self, path: None)

    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()

    lora_path = tmp_path / "lightning.safetensors"
    lora_path.write_text("stub")

    FakeModelManager.lora_calls = []

    pipe = wan_video_holocine.WanVideoHoloCinePipeline.from_pretrained(
        torch_dtype=torch.float16,
        device="cpu",
        model_configs=[
            ModelConfig(path="/tmp/text_encoder.safetensors"),
            ModelConfig(path="/tmp/high_noise.gguf", lora_paths=str(lora_path), lora_alpha=0.75),
            ModelConfig(path="/tmp/low_noise.gguf"),
        ],
        tokenizer_config=ModelConfig(path=str(tokenizer_dir)),
        redirect_common_files=False,
    )

    assert isinstance(pipe, wan_video_holocine.WanVideoHoloCinePipeline)

    assert FakeModelManager.lora_calls
    flattened = []
    for paths, alpha in FakeModelManager.lora_calls:
        if isinstance(paths, (list, tuple)):
            flattened.extend((p, alpha) for p in paths)
        else:
            flattened.append((paths, alpha))
    assert any(p == str(lora_path) and a == 0.75 for p, a in flattened)
