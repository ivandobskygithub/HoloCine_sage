import logging
import math
from typing import Optional

import torch
from diffsynth import save_video
from diffsynth.pipelines.wan_video_holocine import WanVideoHoloCinePipeline, ModelConfig

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
    height: int = 240,
    width: int = 416,
    num_inference_steps: int = 50,

    # --- Block Swap Controls ---
    enable_block_swap: bool = True,
    block_swap_limit_gb: float = 32.0,
    block_swap_size: Optional[int] = None,
    block_swap_stride: Optional[int] = None,
    block_swap_device: str = "cpu",
    block_swap_dtype: Optional[torch.dtype] = None,
    
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
        "num_inference_steps": num_inference_steps,
        "enable_block_swap": enable_block_swap,
        "block_swap_limit_gb": block_swap_limit_gb,
        "block_swap_size": block_swap_size,
        "block_swap_stride": block_swap_stride,
        "block_swap_device": block_swap_device,
        "block_swap_dtype": block_swap_dtype,
    }

    if pipe_kwargs["block_swap_dtype"] is None and hasattr(pipe, "torch_dtype"):
        pipe_kwargs["block_swap_dtype"] = pipe.torch_dtype

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

# --- 1. Load Model (Done once) ---
device = 'cuda'
pipe = WanVideoHoloCinePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device=device,
    model_configs=[
        ModelConfig(
            path="D:/development/HoloCine/checkpoints/Wan2.2-T2V-A14B/umt5-xxl-enc-bf16.safetensors",
            offload_device="cpu",
            offload_dtype=torch.bfloat16,
        ),
        ModelConfig(
            path="D:/development/HoloCine/checkpoints/HoloCine_dit/sparse/Holocine_sparse_high_e4m3_fp8.safetensors",
            offload_device="cpu",
            offload_dtype=torch.float8_e4m3fn,
        ),
        ModelConfig(
            path="D:/development/HoloCine/checkpoints/HoloCine_dit/sparse/Holocine_sparse_low_e4m3_fp8.safetensors",
            offload_device="cpu",
            offload_dtype=torch.float8_e4m3fn,
        ),
        ModelConfig(
            path="D:/development/HoloCine/checkpoints/Wan2.2-T2V-A14B/wan_2.1_vae.safetensors",
            offload_device="cpu",
            offload_dtype=torch.float16,
        ),
    ],
)
pipe.dit.use_sparse_self_attn=True
pipe.dit2.use_sparse_self_attn=True
pipe.enable_vram_management()
pipe.to(device)

blockswap_log_paths = [
    handler.baseFilename
    for handler in pipe.logger.handlers
    if isinstance(handler, logging.FileHandler)
]
if blockswap_log_paths:
    print(f"Block swap diagnostics will be written to {blockswap_log_paths[0]}")
else:
    print("Block swap diagnostics file handler was not detected; logs will stream to stdout only.")

# --- 2. Define Common Parameters ---
scene_negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"


# ===================================================================
#                ✨ How to Use ✨
# ===================================================================

# --- Example 1: Call using Structured Input (Choice 1) ---
# (Auto-calculates shot cuts)
print("\n--- Running Example 1 (Structured Input) ---")
run_inference(
    pipe=pipe,
    negative_prompt=scene_negative_prompt,
    output_path="video2.mp4",
    
    # Choice 1 inputs
    global_caption="The scene is set in a lavish, 1920s Art Deco ballroom during a masquerade party. [character1] is a mysterious woman with a sleek bob, wearing a sequined silver dress and an ornate feather mask. [character2] is a dapper gentleman in a black tuxedo, his face half-hidden by a simple black domino mask. The environment is filled with champagne fountains, a live jazz band, and dancing couples in extravagant costumes. This scene contains 5 shots.",
    shot_captions=[
        "Medium shot of [character1] standing by a pillar, observing the crowd, a champagne flute in her hand.",
        "Close-up of [character2] watching her from across the room, a look of intrigue on his visible features.",
        "Medium shot as [character2] navigates the crowd and approaches [character1], offering a polite bow.",
        "Close-up on [character1]'s eyes through her mask, as they crinkle in a subtle, amused smile.",
        "A stylish medium two-shot of them standing together, the swirling party out of focus behind them, as they begin to converse."

    ],
    num_frames=141
)
