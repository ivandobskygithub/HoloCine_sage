"""Example helper to launch HoloCine full-attention inference with GGUF checkpoints.

Update the configuration block to match your local checkpoint layout.
"""
from __future__ import annotations

import os
from typing import Optional

from HoloCine_inference_full_attention import (
    DEFAULT_NEGATIVE_PROMPT,
    apply_lightning_preset,
    build_pipeline,
    run_inference,
)


CONFIG = {
    "checkpoint_root": os.getenv("HOLOCINE_CHECKPOINT_ROOT", "/path/to/checkpoints"),
    "device": None,  # e.g. "cuda:0"
    "use_quantized": True,
    "quant_suffix": "Q4_K_M",  # e.g. Q5_1 for higher precision builds
    "lightning": "wan2.2",  # choose "wan1.1" or "wan2.2", or set to None to skip
    "lightning_alpha": 1.0,
    # Supply custom Lightning LoRA paths if you do not rely on the defaults.
    # The script will fall back to `<checkpoint_root>/lora/...` when left empty.
    "lightning_paths": [],
    "prompt": (
        "A sweeping drone shot over a futuristic coastal city at golden hour, "
        "neon billboards glinting as air taxis weave between slender skyscrapers."
    ),
    "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
    "num_inference_steps": None,  # defaults to preset recommendation or 50
    "output": "holocine_quantized_lightning.mp4",
    "seed": 42,
    "height": 480,
    "width": 832,
    "tiled": True,
    "fps": 15,
    "quality": 5,
}


def main() -> None:
    checkpoint_root = CONFIG["checkpoint_root"]
    pipe = build_pipeline(
        checkpoint_root=checkpoint_root,
        device=CONFIG["device"],
        use_quantized=CONFIG["use_quantized"],
        quant_suffix=CONFIG["quant_suffix"],
    )

    recommended_steps: Optional[int] = apply_lightning_preset(
        pipe,
        CONFIG["lightning"],
        checkpoint_root=checkpoint_root,
        lora_paths=CONFIG["lightning_paths"],
        alpha=CONFIG["lightning_alpha"],
    )

    num_inference_steps = (
        CONFIG["num_inference_steps"]
        or recommended_steps
        or 50
    )

    run_inference(
        pipe=pipe,
        output_path=CONFIG["output"],
        prompt=CONFIG["prompt"],
        negative_prompt=CONFIG["negative_prompt"],
        num_inference_steps=num_inference_steps,
        seed=CONFIG["seed"],
        height=CONFIG["height"],
        width=CONFIG["width"],
        tiled=CONFIG["tiled"],
        fps=CONFIG["fps"],
        quality=CONFIG["quality"],
    )


if __name__ == "__main__":
    main()
