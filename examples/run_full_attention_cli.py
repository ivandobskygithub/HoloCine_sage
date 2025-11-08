from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

from HoloCine_inference_full_attention import (
    COMBINED_DEMO_NUM_FRAMES,
    COMBINED_DEMO_PROMPT,
    COMBINED_DEMO_SHOT_CUT_FRAMES,
    DEFAULT_NEGATIVE_PROMPT,
    LIGHTNING_PRESETS,
    STRUCTURED_DEMO_GLOBAL_CAPTION,
    STRUCTURED_DEMO_NUM_FRAMES,
    STRUCTURED_DEMO_SHOT_CAPTIONS,
    apply_lightning_preset,
    build_pipeline,
    run_inference,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CLI runner for HoloCine full-attention inference.")
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


def run_demo(
    pipe,
    *,
    output_path: str,
    negative_prompt: str,
    num_inference_steps: int,
    seed: int,
    height: int,
    width: int,
    tiled: bool,
    fps: int,
    quality: int,
) -> None:
    print("\n--- Running Structured Multi-shot Demo ---")
    run_inference(
        pipe=pipe,
        output_path=output_path,
        global_caption=STRUCTURED_DEMO_GLOBAL_CAPTION,
        shot_captions=list(STRUCTURED_DEMO_SHOT_CAPTIONS),
        num_frames=STRUCTURED_DEMO_NUM_FRAMES,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        seed=seed,
        height=height,
        width=width,
        tiled=tiled,
        fps=fps,
        quality=quality,
    )

    base = Path(output_path)
    alt_output = f"{base.stem}_raw{base.suffix}" if base.suffix else f"{output_path}_raw.mp4"

    print("\n--- Running Combined Prompt Demo ---")
    run_inference(
        pipe=pipe,
        output_path=alt_output,
        prompt=COMBINED_DEMO_PROMPT,
        num_frames=COMBINED_DEMO_NUM_FRAMES,
        shot_cut_frames=list(COMBINED_DEMO_SHOT_CUT_FRAMES),
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        seed=seed,
        height=height,
        width=width,
        tiled=tiled,
        fps=fps,
        quality=quality,
    )


def main() -> None:
    args = parse_args()

    pipe = build_pipeline(
        checkpoint_root=args.checkpoint_root,
        device=args.device,
        use_quantized=args.use_quantized,
        quant_suffix=args.quant_suffix,
    )

    recommended_steps: Optional[int] = apply_lightning_preset(
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
            output_path=args.output,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            seed=args.seed,
            height=args.height,
            width=args.width,
            tiled=args.tiled,
            fps=args.fps,
            quality=args.quality,
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
