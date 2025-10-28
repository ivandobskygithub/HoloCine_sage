"""Utility helpers for selecting and executing the attention backend.

This project was originally written with FlashAttention v2/v3 in mind.
The helpers in this file extend the implementation so that alternative
backends such as SageAttention 2++ and SageAttention 3 can be selected
at runtime.  The default behaviour keeps the "best" backend available
while also exposing an environment variable so users can force a
specific implementation if they want to.

The exported helpers are intentionally lightweight and only depend on
PyTorch and einops so that call-sites (e.g. WAN's DiT implementation and
StepVideo's text encoder) do not need to perform their own conditional
imports.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch
import torch.nn.functional as F
from einops import rearrange


class AttentionBackendError(RuntimeError):
    """Raised when a requested backend is not available."""


@dataclass(frozen=True)
class _BackendImplementation:
    """Holds implementation details for a backend."""

    name: str
    runner: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, int, float], torch.Tensor]
    supports_varlen: bool = False


def _load_backends() -> Dict[str, _BackendImplementation]:
    """Discover and prepare all supported backends.

    Returns a mapping of backend identifier -> implementation.  The
    detection is done lazily at module import time so that a missing
    dependency does not crash the import.
    """

    backends: Dict[str, _BackendImplementation] = {}

    flash_attn_v3_func = None
    flash_attn_v3_varlen = None
    try:
        import flash_attn_interface as _flash_attn_interface  # type: ignore

        flash_attn_v3_func = getattr(_flash_attn_interface, "flash_attn_func", None)
        flash_attn_v3_varlen = getattr(_flash_attn_interface, "flash_attn_varlen_func", None)
    except ModuleNotFoundError:
        pass

    flash_attn_v2_func = None
    flash_attn_v2_varlen = None
    try:
        import flash_attn as _flash_attn  # type: ignore

        flash_attn_v2_func = getattr(_flash_attn, "flash_attn_func", None)
        try:
            from flash_attn.flash_attn_interface import (  # type: ignore
                flash_attn_varlen_func as _flash_attn_v2_varlen,
            )

            flash_attn_v2_varlen = _flash_attn_v2_varlen
        except Exception:
            flash_attn_v2_varlen = None
    except ModuleNotFoundError:
        pass

    sage_attn_v3_func = None
    sage_attn_v2pp_func = None
    try:
        import sageattention as _sageattention  # type: ignore

        sage_attn_v3_func = getattr(_sageattention, "flash_attn_func", None)
        # SageAttention 2++ exposes `sageattn` in the same package.
        sage_attn_v2pp_func = getattr(_sageattention, "sageattn", None)
    except ModuleNotFoundError:
        pass

    def _runner_flash_like(attn_func):
        def _run(q, k, v, num_heads, dropout_p=0.0):
            q_local = rearrange(q, "b s (n d) -> b s n d", n=num_heads).contiguous()
            k_local = rearrange(k, "b s (n d) -> b s n d", n=num_heads).contiguous()
            v_local = rearrange(v, "b s (n d) -> b s n d", n=num_heads).contiguous()
            result = attn_func(q_local, k_local, v_local, dropout_p=dropout_p)
            if isinstance(result, tuple):
                result = result[0]
            return rearrange(result, "b s n d -> b s (n d)", n=num_heads)

        return _run

    def _runner_sage2pp(attn_func):
        def _run(q, k, v, num_heads, dropout_p=0.0):  # pylint: disable=unused-argument
            # SageAttention 2++ expects tensors in [B, H, S, D] layout.
            q_local = rearrange(q, "b s (n d) -> b n s d", n=num_heads).contiguous()
            k_local = rearrange(k, "b s (n d) -> b n s d", n=num_heads).contiguous()
            v_local = rearrange(v, "b s (n d) -> b n s d", n=num_heads).contiguous()
            result = attn_func(q_local, k_local, v_local)
            if isinstance(result, tuple):
                result = result[0]
            return rearrange(result, "b n s d -> b s (n d)", n=num_heads)

        return _run

    if callable(flash_attn_v3_func):
        backends["flash3"] = _BackendImplementation(
            name="FlashAttention v3",
            runner=_runner_flash_like(flash_attn_v3_func),
            supports_varlen=callable(flash_attn_v3_varlen),
        )

    if callable(sage_attn_v3_func):
        backends["sage3"] = _BackendImplementation(
            name="SageAttention 3",
            runner=_runner_flash_like(sage_attn_v3_func),
            supports_varlen=False,
        )

    if callable(flash_attn_v2_func):
        backends["flash2"] = _BackendImplementation(
            name="FlashAttention v2",
            runner=_runner_flash_like(flash_attn_v2_func),
            supports_varlen=callable(flash_attn_v2_varlen),
        )

    if callable(sage_attn_v2pp_func):
        backends["sage2pp"] = _BackendImplementation(
            name="SageAttention 2++",
            runner=_runner_sage2pp(sage_attn_v2pp_func),
            supports_varlen=False,
        )

    def _run_torch(q, k, v, num_heads, dropout_p=0.0):
        q_local = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k_local = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v_local = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        out = F.scaled_dot_product_attention(
            q_local,
            k_local,
            v_local,
            attn_mask=None,
            dropout_p=dropout_p if dropout_p > 0 else 0.0,
            is_causal=False,
        )
        return rearrange(out, "b n s d -> b s (n d)", n=num_heads)

    backends["pytorch"] = _BackendImplementation(
        name="PyTorch scaled_dot_product_attention",
        runner=_run_torch,
        supports_varlen=False,
    )

    return backends


_BACKENDS = _load_backends()
_SELECTED_BACKEND: Optional[_BackendImplementation] = None


def _normalise_backend_name(name: str) -> str:
    return name.lower().replace("-", "").replace("_", "")


def _preferred_backend_from_env() -> Optional[str]:
    env_value = os.environ.get("HOLOCINE_ATTENTION_BACKEND")
    if env_value is None:
        return None
    normalised = _normalise_backend_name(env_value)
    if normalised == "auto":
        return None
    return normalised


def _select_backend(preferred: Optional[str] = None) -> _BackendImplementation:
    preferred = preferred or _preferred_backend_from_env()

    candidates = [
        "flash3",
        "sage3",
        "flash2",
        "sage2pp",
        "pytorch",
    ]

    if preferred is not None:
        preferred = _normalise_backend_name(preferred)
        if preferred not in _BACKENDS:
            raise AttentionBackendError(
                f"Attention backend '{preferred}' is not available on this system."
            )
        return _BACKENDS[preferred]

    for candidate in candidates:
        backend = _BACKENDS.get(candidate)
        if backend is not None:
            return backend

    raise AttentionBackendError("No usable attention backend found.")


def _ensure_backend_loaded() -> None:
    global _SELECTED_BACKEND  # noqa: PLW0603
    if _SELECTED_BACKEND is None:
        _SELECTED_BACKEND = _select_backend()


def get_attention_backend_name() -> str:
    """Return the human readable name of the currently selected backend."""

    _ensure_backend_loaded()
    assert _SELECTED_BACKEND is not None  # for type checkers
    return _SELECTED_BACKEND.name


def run_attention_backend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: int,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """Execute the attention kernel for the selected backend."""

    _ensure_backend_loaded()
    assert _SELECTED_BACKEND is not None
    return _SELECTED_BACKEND.runner(q, k, v, num_heads, dropout_p)


def backend_supports_varlen_attention() -> bool:
    """Whether the current backend has an efficient varlen attention kernel."""

    _ensure_backend_loaded()
    assert _SELECTED_BACKEND is not None
    return _SELECTED_BACKEND.supports_varlen


def get_varlen_attention_func() -> Optional[Callable]:
    """Return the raw varlen attention function if available."""

    _ensure_backend_loaded()
    assert _SELECTED_BACKEND is not None

    # The discovery function stores raw callables on the module level to
    # avoid importing optional packages here again.
    name = _normalise_backend_name(_SELECTED_BACKEND.name)

    if name == "flashattentionv3":
        try:
            import flash_attn_interface as _flash_attn_interface  # type: ignore

            return getattr(_flash_attn_interface, "flash_attn_varlen_func", None)
        except ModuleNotFoundError:
            return None

    if name == "flashattentionv2":
        try:
            from flash_attn.flash_attn_interface import (  # type: ignore
                flash_attn_varlen_func as _flash_attn_v2_varlen,
            )

            return _flash_attn_v2_varlen
        except Exception:
            return None

    return None


__all__ = [
    "AttentionBackendError",
    "backend_supports_varlen_attention",
    "get_attention_backend_name",
    "get_varlen_attention_func",
    "run_attention_backend",
]

