from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import torch
from diffsynth import save_video
from diffsynth.pipelines.wan_video_holocine import WanVideoHoloCinePipeline, ModelConfig


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
    "dit_high_fp8": "HoloCine_dit/full/Holocine_full_high_e4m3_fp8.safetensors",
    "dit_low_fp8": "HoloCine_dit/full/Holocine_full_low_e4m3_fp8.safetensors",
    "dit_high_quant": "HoloCine_dit/quantized/HoloCine-Full-HighNoise-{suffix}.gguf",
    "dit_low_quant": "HoloCine_dit/quantized/HoloCine-Full-LowNoise-{suffix}.gguf",
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
    if latent_storage_dtype is None and hasattr(torch, "float8_e4m3fn"):
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
    print(f"Running inference...")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HoloCine video generation.")
    parser.add_argument("--checkpoint-root", default=os.getenv("HOLOCINE_CHECKPOINT_ROOT"))
    parser.add_argument("--device")
    parser.add_argument("--use-quantized", action="store_true")
    parser.add_argument("--quant-suffix", default="Q4_K_M")
    parser.add_argument("--lightning", choices=sorted(LIGHTNING_PRESETS.keys()))
    parser.add_argument("--lightning-alpha", type=float, default=1.0)
    parser.add_argument("--lightning-path", action="append", default=[])
    parser.add_argument("--num-inference-steps", type=int)
    parser.add_argument("--output", default="video1.mp4")
    parser.add_argument("--prompt")
    parser.add_argument("--global-caption")
    parser.add_argument("--shot", action="append", default=[])
    parser.add_argument("--num-frames", type=int)
    parser.add_argument("--shot-cut-frame", action="append", type=int)
    parser.add_argument("--negative-prompt")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--quality", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--tiled", dest="tiled", action="store_true")
    parser.add_argument("--no-tiled", dest="tiled", action="store_false")
    parser.set_defaults(tiled=True)
    parser.add_argument("--demo", action="store_true", help="Run the built-in showcase prompts.")
    return parser.parse_args()


def run_demo(pipe: WanVideoHoloCinePipeline, num_inference_steps: int, output_path: str, negative_prompt: str) -> None:
    print("\n--- Running Example 1 (Structured Input) ---")
    run_inference(
        pipe=pipe,
        negative_prompt=negative_prompt,
        output_path=output_path,
        global_caption=(
            "The scene is set in a lavish, 1920s Art Deco ballroom during a masquerade party. "
            "[character1] is a mysterious woman with a sleek bob, wearing a sequined silver dress and an ornate feather mask. "
            "[character2] is a dapper gentleman in a black tuxedo, his face half-hidden by a simple black domino mask. "
            "The environment is filled with champagne fountains, a live jazz band, and dancing couples in extravagant costumes. "
            "This scene contains 5 shots."
        ),
        shot_captions=[
            "Medium shot of [character1] standing by a pillar, observing the crowd, a champagne flute in her hand.",
            "Close-up of [character2] watching her from across the room, a look of intrigue on his visible features.",
            "Medium shot as [character2] navigates the crowd and approaches [character1], offering a polite bow.",
            "Close-up on [character1]'s eyes through her mask, as they crinkle in a subtle, amused smile.",
            "A stylish medium two-shot of them standing together, the swirling party out of focus behind them, as they begin to converse.",
        ],
        num_frames=81,
        num_inference_steps=num_inference_steps,
    )


def main() -> None:
    args = parse_args()

    pipe = build_pipeline(
        checkpoint_root=args.checkpoint_root,
        device=args.device,
        use_quantized=args.use_quantized,
        quant_suffix=args.quant_suffix,
    )

    recommended_steps = apply_lightning_preset(
        pipe,
        args.lightning,
        checkpoint_root=args.checkpoint_root,
        lora_paths=args.lightning_path,
        alpha=args.lightning_alpha,
    )

    num_inference_steps = args.num_inference_steps or recommended_steps or 50
    negative_prompt = args.negative_prompt or DEFAULT_NEGATIVE_PROMPT

    if args.demo or (not args.prompt and not args.global_caption and not args.shot):
        run_demo(
            pipe,
            num_inference_steps=num_inference_steps,
            output_path=args.output,
            negative_prompt=negative_prompt,
        )
        return

    run_inference(
        pipe=pipe,
        output_path=args.output,
        global_caption=args.global_caption,
        shot_captions=args.shot or None,
        prompt=args.prompt,
        negative_prompt=negative_prompt,
        num_frames=args.num_frames,
        shot_cut_frames=args.shot_cut_frame,
        seed=args.seed,
        tiled=args.tiled,
        height=args.height,
        width=args.width,
        num_inference_steps=num_inference_steps,
        fps=args.fps,
        quality=args.quality,
    )


if __name__ == "__main__":
    main()
