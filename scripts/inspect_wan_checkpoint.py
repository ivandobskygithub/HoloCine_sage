#!/usr/bin/env python3
"""Utility for inspecting WAN checkpoints.

This helper loads a checkpoint with ``diffsynth.models.utils.load_state_dict``
and runs the extracted tensors through ``diffsynth.models.model_manager._infer_wan_kwargs``
so we can see the configuration parameters that would be used to construct a
:class:`WanModel` instance.  It works for both FP8 safetensors and GGUF files as
long as the ``gguf`` dependency is available.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import torch

from diffsynth.models import model_manager, utils as model_utils


def _format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, tuple)):
        inner = ", ".join(_format_value(item) for item in value)
        bracket = "[]" if isinstance(value, list) else "()"
        return f"{bracket[0]}{inner}{bracket[1]}"
    return str(value)


def inspect_checkpoint(path: Path) -> None:
    path = path.expanduser().resolve()
    print(f"\nInspecting {path}...")
    if not path.exists():
        print("  ! File not found.")
        return
    try:
        state_dict = model_utils.load_state_dict(
            str(path), torch_dtype=torch.float32, device="cpu"
        )
    except Exception as exc:  # pragma: no cover - forwarding for interactive usage
        print(f"  ! Failed to load state dict: {exc}")
        return
    inferred = model_manager._infer_wan_kwargs(state_dict)
    for key in sorted(inferred):
        value = inferred[key]
        print(f"  {key:>28}: {_format_value(value)}")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoints",
        nargs="+",
        type=Path,
        help="Paths to WanModel checkpoints (FP8 safetensors or GGUF).",
    )
    args = parser.parse_args(argv)

    for path in args.checkpoints:
        inspect_checkpoint(path)

    return 0


if __name__ == "__main__":  # pragma: no cover - manual utility
    raise SystemExit(main())
