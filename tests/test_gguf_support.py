import sys

import numpy as np
import torch

import HoloCine_inference_full_attention as full_attention
from diffsynth.models.utils import load_state_dict
from diffsynth.pipelines.wan_video_holocine import WanVideoHoloCinePipeline


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
            self.tensors = [FakeTensor(b"model.weight")]

    monkeypatch.setitem(sys.modules, "gguf", type("M", (), {"GGUFReader": FakeGGUFReader}))

    file_path = tmp_path / "dummy.gguf"
    file_path.write_bytes(b"fake")

    state_dict = load_state_dict(str(file_path))
    assert "model.weight" in state_dict
    (key,) = state_dict.keys()
    assert isinstance(key, str)
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
    configs = full_attention.build_model_configs(
        str(tmp_path),
        use_quantized=True,
        quant_suffix="Q4_K_M",
    )
    assert any(path.endswith("Q4_K_M.gguf") for path in (cfg.path for cfg in configs))


def test_prepare_multishot_inputs_matches_structured_prompt(monkeypatch):
    class DummyPipe:
        def __init__(self):
            self.calls = []

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            return "video"

    pipe = DummyPipe()

    saved = {}

    def fake_save_video(video, output_path, fps, quality):
        saved.update(
            {
                "video": video,
                "output_path": output_path,
                "fps": fps,
                "quality": quality,
            }
        )

    monkeypatch.setattr(full_attention, "save_video", fake_save_video)

    base_caption = full_attention.STRUCTURED_DEMO_GLOBAL_CAPTION.replace(
        " This scene contains 5 shots.", ""
    )
    shot_captions = list(full_attention.STRUCTURED_DEMO_SHOT_CAPTIONS)
    num_frames = full_attention.STRUCTURED_DEMO_NUM_FRAMES

    expected_inputs = full_attention.prepare_multishot_inputs(
        global_caption=base_caption,
        shot_captions=shot_captions,
        total_frames=num_frames,
    )

    full_attention.run_inference(
        pipe=pipe,
        output_path="structured.mp4",
        global_caption=base_caption,
        shot_captions=shot_captions,
        num_frames=num_frames,
        negative_prompt=full_attention.DEFAULT_NEGATIVE_PROMPT,
        num_inference_steps=28,
    )

    assert pipe.calls, "Pipeline was not invoked"
    kwargs = pipe.calls[-1]
    assert kwargs["prompt"] == expected_inputs["prompt"]
    assert kwargs["num_frames"] == expected_inputs["num_frames"]
    assert kwargs["shot_cut_frames"] == expected_inputs["shot_cut_frames"]
    assert saved["output_path"] == "structured.mp4"
    assert saved["fps"] == 15
    assert saved["quality"] == 5


def test_run_inference_respects_combined_prompt(monkeypatch):
    class DummyPipe:
        def __init__(self):
            self.calls = []

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            return "video"

    pipe = DummyPipe()

    monkeypatch.setattr(full_attention, "save_video", lambda *args, **kwargs: None)

    full_attention.run_inference(
        pipe=pipe,
        output_path="combined.mp4",
        prompt=full_attention.COMBINED_DEMO_PROMPT,
        num_frames=full_attention.COMBINED_DEMO_NUM_FRAMES,
        shot_cut_frames=list(full_attention.COMBINED_DEMO_SHOT_CUT_FRAMES),
        negative_prompt=full_attention.DEFAULT_NEGATIVE_PROMPT,
        num_inference_steps=32,
    )

    assert pipe.calls, "Pipeline was not invoked"
    kwargs = pipe.calls[-1]
    assert kwargs["prompt"] == full_attention.COMBINED_DEMO_PROMPT
    assert kwargs["num_frames"] == full_attention.COMBINED_DEMO_NUM_FRAMES
    assert kwargs["shot_cut_frames"] == list(
        full_attention.COMBINED_DEMO_SHOT_CUT_FRAMES
    )
