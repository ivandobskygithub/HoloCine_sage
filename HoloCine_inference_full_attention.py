from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import torch
from diffsynth import save_video
from diffsynth.pipelines.wan_video_holocine import ModelConfig, WanVideoHoloCinePipeline


@dataclass(frozen=True)
class LightningPreset:
    name: str
    env_var: str
    relative_paths: Sequence[str]
    default_steps: int = 8
    alpha: float = 1.0


DEFAULT_CHECKPOINT_LAYOUT = {
    "text_encoder": "Wan2.2-T2V-A14B/umt5-xxl-enc-bf16.safetensors",
    "vae": "Wan2.2-T2V-A14B/wan_2.1_vae.safetensors",
    "dit_high_fp8": "full/Holocine_full_high_e4m3_fp8.safetensors",
    "dit_low_fp8": "full/Holocine_full_low_e4m3_fp8.safetensors",
    "dit_high_quant": "quantized/HoloCine-Full-HighNoise-{suffix}.gguf",
    "dit_low_quant": "quantized/HoloCine-Full-LowNoise-{suffix}.gguf",
}


LIGHTNING_PRESETS = {
    "wan1.1": LightningPreset(
        name="wan1.1",
        env_var="HOLOCINE_WAN11_LIGHTNING_LORA",
        relative_paths=("lora/Wan1.1-Lightning-8step.safetensors",),
    ),
    "wan2.2": LightningPreset(
        name="wan2.2",
        env_var="HOLOCINE_WAN22_LIGHTNING_LORA",
        relative_paths=("lora/Wan2.2-Lightning-8step.safetensors",),
    ),
}


STRUCTURED_DEMO_GLOBAL_CAPTION = (
    "The scene is set in a lavish, 1920s Art Deco ballroom during a masquerade party. "
    "[character1] is a mysterious woman with a sleek bob, wearing a sequined silver dress and an ornate feather mask. "
    "[character2] is a dapper gentleman in a black tuxedo, his face half-hidden by a simple black domino mask. "
    "The environment is filled with champagne fountains, a live jazz band, and dancing couples in extravagant costumes. "
    "This scene contains 5 shots."
)

STRUCTURED_DEMO_SHOT_CAPTIONS = (
    "Medium shot of [character1] standing by a pillar, observing the crowd, a champagne flute in her hand.",
    "Close-up of [character2] watching her from across the room, a look of intrigue on his visible features.",
    "Medium shot as [character2] navigates the crowd and approaches [character1], offering a polite bow.",
    "Close-up on [character1]'s eyes through her mask, as they crinkle in a subtle, amused smile.",
    "A stylish medium two-shot of them standing together, the swirling party out of focus behind them, as they begin to converse.",
)

STRUCTURED_DEMO_NUM_FRAMES = 81

COMBINED_DEMO_PROMPT = (
    "[global caption] The scene features a young painter, [character1], with paint-smudged cheeks and intense, focused eyes. "
    "Her hair is tied up messily. The setting is a bright, sun-drenched art studio with large windows, canvases, and the smell "
    "of oil paint. This scene contains 6 shots. [per shot caption] Medium shot of [character1] standing back from a large "
    "canvas, brush in hand, critically observing her work. [shot cut] Close-up of her hand holding the brush, dabbing it "
    "thoughtfully onto a palette of vibrant colors. [shot cut] Extreme close-up of her eyes, narrowed in concentration as she "
    "studies the canvas. [shot cut] Close-up on the canvas, showing a detailed, textured brushstroke being slowly applied. "
    "[shot cut] Medium close-up of [character1]'s face, a small, satisfied smile appears as she finds the right color. [shot cut] "
    "Over-the-shoulder shot showing her add a final, delicate highlight to the painting."
)

COMBINED_DEMO_NUM_FRAMES = 241
COMBINED_DEMO_SHOT_CUT_FRAMES = (37, 73, 113, 169, 205)


def _resolve_checkpoint_path(checkpoint_root: Optional[str], path: str) -> str:
    if os.path.isabs(path):
        return os.path.expanduser(path)
    root = checkpoint_root or os.getenv("HOLOCINE_CHECKPOINT_ROOT", "")
    candidate = os.path.expanduser(path)
    if root:
        return os.path.join(root, candidate)
    return candidate


def build_model_configs(
    checkpoint_root: Optional[str],
    *,
    use_quantized: bool,
    quant_suffix: str,
    overrides: Optional[dict[str, str]] = None,
) -> list[ModelConfig]:
    overrides = overrides or {}
    layout = DEFAULT_CHECKPOINT_LAYOUT.copy()
    layout.update(overrides)

    high_key = "dit_high_quant" if use_quantized else "dit_high_fp8"
    low_key = "dit_low_quant" if use_quantized else "dit_low_fp8"

    high_path_template = layout[high_key]
    low_path_template = layout[low_key]

    def format_path(template: str) -> str:
        if "{suffix}" in template:
            template = template.format(suffix=quant_suffix)
        return _resolve_checkpoint_path(checkpoint_root, template)

    text_encoder_path = format_path(layout["text_encoder"])
    vae_path = format_path(layout["vae"])
    dit_high_path = format_path(high_path_template)
    dit_low_path = format_path(low_path_template)

    configs = [
        ModelConfig(
            path=text_encoder_path,
            offload_device="cpu",
            offload_dtype=torch.bfloat16,
        ),
        ModelConfig(
            path=dit_high_path,
            offload_device="cpu",
            offload_dtype=torch.float16 if use_quantized else getattr(torch, "float8_e4m3fn", torch.bfloat16),
        ),
        ModelConfig(
            path=dit_low_path,
            offload_device="cpu",
            offload_dtype=torch.float16 if use_quantized else getattr(torch, "float8_e4m3fn", torch.bfloat16),
        ),
        ModelConfig(
            path=vae_path,
            offload_device="cpu",
            offload_dtype=torch.float16,
        ),
    ]
    return configs


def apply_lightning_preset(
    pipe: WanVideoHoloCinePipeline,
    preset_name: Optional[str],
    *,
    checkpoint_root: Optional[str] = None,
    lora_paths: Optional[Iterable[str]] = None,
    alpha: float = 1.0,
) -> Optional[int]:
    if preset_name is None and not lora_paths:
        return None

    resolved_paths: list[str] = []
    if lora_paths:
        resolved_paths.extend(os.path.expanduser(p) for p in lora_paths)

    preset = LIGHTNING_PRESETS.get(preset_name or "") if preset_name else None
    if preset is not None:
        env_value = os.getenv(preset.env_var)
        if env_value:
            for candidate in env_value.split(os.pathsep):
                candidate = candidate.strip()
                if candidate:
                    resolved_paths.append(os.path.expanduser(candidate))
        if not env_value:
            for relative in preset.relative_paths:
                resolved_paths.append(_resolve_checkpoint_path(checkpoint_root, relative))

    if not resolved_paths:
        raise ValueError(
            "No LoRA paths provided for Lightning preset. Provide a path via the "
            "--lightning-path flag or set the corresponding environment variable."
        )

    pipe.apply_lightning_lora(resolved_paths, alpha=alpha)
    return preset.default_steps if preset is not None else None


def build_pipeline(
    *,
    checkpoint_root: Optional[str] = None,
    device: Optional[str] = None,
    use_quantized: bool = False,
    quant_suffix: str = "Q4_K_M",
    model_overrides: Optional[dict[str, str]] = None,
    torch_dtype: torch.dtype = torch.bfloat16,
    latent_storage_dtype: Optional[torch.dtype] = None,
) -> WanVideoHoloCinePipeline:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if latent_storage_dtype is None and hasattr(torch, "float8_e4m3fn") and not use_quantized:
        latent_storage_dtype = getattr(torch, "float8_e4m3fn")

    configs = build_model_configs(
        checkpoint_root,
        use_quantized=use_quantized,
        quant_suffix=quant_suffix,
        overrides=model_overrides,
    )

    pipe = WanVideoHoloCinePipeline.from_pretrained(
        torch_dtype=torch_dtype,
        latent_storage_dtype=latent_storage_dtype,
        device=device,
        model_configs=configs,
    )
    pipe.enable_vram_management()
    pipe.to(device)
    return pipe

# ---------------------------------------------------
#                Helper Functions
# --------------------------------------------------- 

def enforce_4t_plus_1(n: int) -> int:
    """Forces an integer 'n' to the closest 4t+1 form."""
    t = round((n - 1) / 4)
    return 4 * t + 1

def prepare_multishot_inputs(
    global_caption: str,
    shot_captions: list[str],
    total_frames: int,
    custom_shot_cut_frames: list[int] = None
) -> dict:
    """
    (Helper for Mode 1)
    Prepares the inference parameters from user-friendly segmented inputs.
    """
    
    num_shots = len(shot_captions)
    
    # 1. Prepare 'prompt'
    if "This scene contains" not in global_caption:
        global_caption = global_caption.strip() + f" This scene contains {num_shots} shots."
    per_shot_string = " [shot cut] ".join(shot_captions)
    prompt = f"[global caption] {global_caption} [per shot caption] {per_shot_string}"

    # 2. Prepare 'num_frames'
    processed_total_frames = enforce_4t_plus_1(total_frames)

    # 3. Prepare 'shot_cut_frames'
    num_cuts = num_shots - 1
    processed_shot_cuts = []

    if custom_shot_cut_frames:
        # User provided custom cuts
        print(f"Using {len(custom_shot_cut_frames)} user-defined shot cuts (enforcing 4t+1).")
        for frame in custom_shot_cut_frames:
            processed_shot_cuts.append(enforce_4t_plus_1(frame))
    else:
        # Auto-calculate cuts
        print(f"Auto-calculating {num_cuts} shot cuts.")
        if num_cuts > 0:
            ideal_step = processed_total_frames / num_shots
            for i in range(1, num_shots):
                approx_cut_frame = i * ideal_step
                processed_shot_cuts.append(enforce_4t_plus_1(round(approx_cut_frame)))

    processed_shot_cuts = sorted(list(set(processed_shot_cuts)))
    processed_shot_cuts = [f for f in processed_shot_cuts if f > 0 and f < processed_total_frames]

    return {
        "prompt": prompt,
        "shot_cut_frames": processed_shot_cuts,
        "num_frames": processed_total_frames
    }

# ---------------------------------------------------
# 
#           ✨ Main Inference Wrapper ✨
#
# ---------------------------------------------------

def run_inference(
    pipe: WanVideoHoloCinePipeline,
    output_path: str,
    
    # --- Prompting Options (Auto-detect) ---
    global_caption: str = None,
    shot_captions: list[str] = None,
    prompt: str = None,
    negative_prompt: str = None,
    
    # --- Core Generation Parameters (All Optional) ---
    num_frames: int = None,
    shot_cut_frames: list[int] = None,
    
    # --- Other Generation Parameters ---
    seed: int = 0,
    tiled: bool = True,
    height: int = 480,
    width: int = 832,
    num_inference_steps: int = 50,
    
    # --- Output Parameters ---
    fps: int = 15,
    quality: int = 5
):
    """
    Runs the inference pipeline, auto-detecting the input mode
    and honoring pipeline defaults for optional parameters.
    
    Mode 1 (Structured): Provide 'global_caption', 'shot_captions', 'num_frames'.
                         'shot_cut_frames' is optional (auto-calculated).
    Mode 2 (Raw): Provide 'prompt'.
                  'num_frames' and 'shot_cut_frames' are optional.
    """
    
    # --- 1. Prepare 'pipe_kwargs' dictionary ---
    pipe_kwargs = {
        "negative_prompt": negative_prompt,
        "seed": seed,
        "tiled": tiled,
        "height": height,
        "width": width,
        "num_inference_steps": num_inference_steps
    }

    # --- 2. Auto-Detection Logic ---
    if global_caption and shot_captions:
        # --- Mode 1: Structured Input ---
        print("--- Detected Structured Input (Mode 1) ---")
        if num_frames is None:
            raise ValueError("Must provide 'num_frames' for structured input (Mode 1).")
        
        # Use the helper function
        inputs = prepare_multishot_inputs(
            global_caption=global_caption,
            shot_captions=shot_captions,
            total_frames=num_frames,
            custom_shot_cut_frames=shot_cut_frames
        )
        pipe_kwargs.update(inputs)

    elif prompt:
        # --- Mode 2: Raw String Input ---
        print("--- Detected Raw String Input (Mode 2) ---")
        pipe_kwargs["prompt"] = prompt
        
        # Process num_frames ONLY if provided
        if num_frames is not None:
            processed_frames = enforce_4t_plus_1(num_frames)
            if num_frames != processed_frames:
                print(f"Corrected 'num_frames': {num_frames} -> {processed_frames}")
            pipe_kwargs["num_frames"] = processed_frames
        else:
            print("Using default 'num_frames' (if any).")
            pipe_kwargs["num_frames"] = None
        
        # Process shot_cut_frames ONLY if provided
        if shot_cut_frames is not None:
            processed_cuts = [enforce_4t_plus_1(f) for f in shot_cut_frames]
            if shot_cut_frames != processed_cuts:
                print(f"Corrected 'shot_cut_frames': {shot_cut_frames} -> {processed_cuts}")
            pipe_kwargs["shot_cut_frames"] = processed_cuts
        else:
            print("Using default 'shot_cut_frames' (if any).")
            pipe_kwargs["shot_cut_frames"] = None
        
    else:
        raise ValueError("Invalid inputs. Provide either (global_caption, shot_captions, num_frames) OR (prompt).")

    # --- 3. Filter out None values before calling pipe ---
    # This ensures we don't pass 'num_frames=None' and override a 
    # default value (e.g., num_frames=25) inside the pipeline.
    final_pipe_kwargs = {k: v for k, v in pipe_kwargs.items() if v is not None}
    
    if "prompt" not in final_pipe_kwargs:
        raise ValueError("A 'prompt' or ('global_caption' + 'shot_captions') is required.")

    # --- 4. Run Generation ---
    print("Running inference...")
    if "num_frames" in final_pipe_kwargs:
        print(f"  Total frames: {final_pipe_kwargs['num_frames']}")
    if "shot_cut_frames" in final_pipe_kwargs:
        print(f"  Cuts: {final_pipe_kwargs['shot_cut_frames']}")

    video = pipe(**final_pipe_kwargs)
    
    save_video(video, output_path, fps=fps, quality=quality)
    print(f"Video saved successfully to {output_path}")


# ---------------------------------------------------
#
#                 Script Execution
#
# ---------------------------------------------------

DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，"
    "JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
    "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


# ---------------------------------------------------
#             Default Configuration Values
# ---------------------------------------------------

CHECKPOINT_ROOT = os.getenv("HOLOCINE_CHECKPOINT_ROOT", "D:/development/HoloCine_sage/models/")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_QUANTIZED = True  # Switch to True to enable QuantStack GGUF checkpoints
QUANT_SUFFIX = "Q4_K_M"
MODEL_OVERRIDES: dict[str, str] = {}

LIGHTNING_PRESET: Optional[str] = None  # choose "wan1.1" or "wan2.2" to enable Lightning
LIGHTNING_ALPHA: float = 1.0
LIGHTNING_PATHS: Sequence[str] = ()

NUM_INFERENCE_STEPS: Optional[int] = None
HEIGHT = 480
WIDTH = 832
TILED = True
SEED = 0
FPS = 15
QUALITY = 5


def load_full_attention_pipeline() -> WanVideoHoloCinePipeline:
    """Load the preconfigured full-attention pipeline."""

    print("Loading HoloCine pipeline...")
    pipe = build_pipeline(
        checkpoint_root=CHECKPOINT_ROOT,
        device=DEVICE,
        use_quantized=USE_QUANTIZED,
        quant_suffix=QUANT_SUFFIX,
        model_overrides=MODEL_OVERRIDES or None,
    )
    return pipe


def _describe_configuration(*, steps: int) -> None:
    print("\n--- HoloCine Full Attention Configuration ---")
    print(f"Device: {DEVICE}")
    print(f"Quantized GGUF: {'enabled' if USE_QUANTIZED else 'disabled'}")
    if USE_QUANTIZED:
        print(f"  Quant suffix: {QUANT_SUFFIX}")
    if LIGHTNING_PRESET or LIGHTNING_PATHS:
        print(f"Lightning preset: {LIGHTNING_PRESET or 'custom paths'} (alpha={LIGHTNING_ALPHA})")
    else:
        print("Lightning preset: none")
    print(f"Inference steps: {steps}")
    print("--------------------------------------------\n")


def _run_structured_demo(pipe: WanVideoHoloCinePipeline, *, steps: int) -> None:
    print("\n--- Running Example 1 (Structured Input) ---")
    run_inference(
        pipe=pipe,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        output_path="video1.mp4",
        global_caption=STRUCTURED_DEMO_GLOBAL_CAPTION,
        shot_captions=list(STRUCTURED_DEMO_SHOT_CAPTIONS),
        num_frames=STRUCTURED_DEMO_NUM_FRAMES,
        num_inference_steps=steps,
        seed=SEED,
        tiled=TILED,
        height=HEIGHT,
        width=WIDTH,
        fps=FPS,
        quality=QUALITY,
    )


if __name__ == "__main__":
    pipeline = load_full_attention_pipeline()

    recommended_steps: Optional[int] = None
    if LIGHTNING_PRESET or LIGHTNING_PATHS:
        recommended_steps = apply_lightning_preset(
            pipeline,
            LIGHTNING_PRESET,
            checkpoint_root=CHECKPOINT_ROOT,
            lora_paths=LIGHTNING_PATHS,
            alpha=LIGHTNING_ALPHA,
        )

    active_steps = NUM_INFERENCE_STEPS or recommended_steps or 50
    _describe_configuration(steps=active_steps)

    _run_structured_demo(pipeline, steps=active_steps)

    # --- Example 2: Call using Raw String Input (Choice 2) ---
    # Uncomment to try the combined caption format with manual cuts.
    # print("\n--- Running Example 2 (Raw String Input) ---")
    # run_inference(
    #     pipe=pipeline,
    #     negative_prompt=DEFAULT_NEGATIVE_PROMPT,
    #     output_path="video2.mp4",
    #     prompt=COMBINED_DEMO_PROMPT,
    #     num_frames=COMBINED_DEMO_NUM_FRAMES,
    #     shot_cut_frames=list(COMBINED_DEMO_SHOT_CUT_FRAMES),
    #     num_inference_steps=active_steps,
    #     seed=SEED,
    #     tiled=TILED,
    #     height=HEIGHT,
    #     width=WIDTH,
    #     fps=FPS,
    #     quality=QUALITY,
    # )
