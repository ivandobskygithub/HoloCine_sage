"""Utilities for reading GGUF checkpoints.

The quantised HoloCine checkpoints hosted by QuantStack are distributed in
`gguf` format.  The upstream project (``llama.cpp``) ships a small Python
reader that exposes a :class:`GGUFReader` capable of de-quantising tensors to
NumPy arrays.  Importing that package directly is inconvenient for users of
this repository so we provide a thin compatibility layer that attempts to
import the reader from either ``gguf`` or ``llama_cpp``.  The helper falls back
to a descriptive error message when neither backend is available.

Only a subset of the ``GGUFReader`` API is required for our purposes – we only
need to iterate over tensors and convert them to ``torch.Tensor`` objects.  The
logic is intentionally defensive to accommodate minor variations across reader
implementations (attribute names such as ``tensors`` vs ``tensor_infos`` or
``data`` vs ``to_numpy``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, Tuple

import numpy as np
import torch


class GGUFImportError(ImportError):
    """Raised when no compatible GGUF reader implementation can be imported."""


@dataclass
class _TensorHandle:
    """Simple container exposing a tensor ``name`` and ``data`` attribute."""

    name: str
    data: np.ndarray


def _normalise_tensor_name(name) -> str:
    """Convert tensor identifiers returned by GGUF readers to plain strings."""

    if isinstance(name, bytes):
        return name.decode("utf-8", errors="replace")
    if isinstance(name, np.generic):
        name = name.item()
    if not isinstance(name, str):
        name = str(name)
    return name


def _resolve_reader() -> type:
    """Return a ``GGUFReader`` implementation.

    We try importing from ``gguf`` first (which ships with upstream llama.cpp)
    and fall back to the copy bundled with ``llama-cpp-python``.  Both expose an
    identical public API for the features we rely upon.  A descriptive error is
    raised when neither implementation can be imported so that callers can
    provide actionable feedback to users.
    """

    backend_candidates = ("gguf", "llama_cpp")
    for module_name in backend_candidates:
        try:
            module = __import__(module_name, fromlist=["GGUFReader"])
        except ImportError:
            continue
        reader = getattr(module, "GGUFReader", None)
        if reader is not None:
            return reader
    raise GGUFImportError(
        "Unable to import a GGUF reader. Install either 'gguf' from llama.cpp "
        "or 'llama-cpp-python' so quantised checkpoints can be loaded."
    )


def _extract_numpy_array(tensor_obj) -> np.ndarray:
    """Convert a tensor object returned by ``GGUFReader`` to ``np.ndarray``."""

    if hasattr(tensor_obj, "data"):
        data = tensor_obj.data
        if callable(data):
            data = data()
    elif hasattr(tensor_obj, "to_numpy"):
        data = tensor_obj.to_numpy()
    elif hasattr(tensor_obj, "to_ndarray"):
        data = tensor_obj.to_ndarray()
    elif hasattr(tensor_obj, "ndarray"):
        data = tensor_obj.ndarray
    elif hasattr(tensor_obj, "array"):
        data = tensor_obj.array
    else:
        raise TypeError(f"Cannot extract tensor data from {type(tensor_obj)!r}.")

    if isinstance(data, memoryview):
        data = np.frombuffer(data, dtype=np.float32)
    elif isinstance(data, np.ndarray):
        # ``astype`` with ``copy=False`` is a no-op when dtype already matches.
        data = data.astype(np.float32, copy=False)
    else:
        data = np.asarray(data, dtype=np.float32)

    return data


def _iter_reader_tensors(reader) -> Iterator[_TensorHandle]:
    """Iterate over tensors inside ``GGUFReader`` regardless of backend."""

    tensors = getattr(reader, "tensors", None)
    if isinstance(tensors, dict):
        for name, tensor_obj in tensors.items():
            yield _TensorHandle(
                name=_normalise_tensor_name(name),
                data=_extract_numpy_array(tensor_obj),
            )
        return
    if isinstance(tensors, Iterable):
        for tensor_obj in tensors:
            name = getattr(tensor_obj, "name", None)
            if name is None and isinstance(tensor_obj, Tuple) and tensor_obj:
                name = tensor_obj[0]
                tensor_obj = tensor_obj[1]
            if name is None:
                raise AttributeError("GGUF tensor is missing a name attribute.")
            yield _TensorHandle(
                name=_normalise_tensor_name(name),
                data=_extract_numpy_array(tensor_obj),
            )
        return

    tensor_infos = getattr(reader, "tensor_infos", None)
    if tensor_infos is not None:
        get_tensor = getattr(reader, "get_tensor", None)
        if get_tensor is None:
            raise AttributeError(
                "GGUF reader exposes 'tensor_infos' but not 'get_tensor'."
            )
        for info in tensor_infos:
            tensor_obj = get_tensor(info)
            name = getattr(info, "name", None) or getattr(tensor_obj, "name", None)
            if name is None:
                raise AttributeError("Unable to determine tensor name from GGUF reader.")
            yield _TensorHandle(
                name=_normalise_tensor_name(name),
                data=_extract_numpy_array(tensor_obj),
            )
        return

    raise AttributeError("Unknown GGUF reader layout; no tensors available.")


def load_gguf_state_dict(
    file_path: str,
    *,
    torch_dtype: torch.dtype | None = None,
    device: torch.device | str = "cpu",
) -> Dict[str, torch.Tensor]:
    """Load tensors from a GGUF checkpoint and return a PyTorch state dict."""

    reader_cls = _resolve_reader()
    reader = reader_cls(file_path)
    state_dict: Dict[str, torch.Tensor] = {}

    for tensor in _iter_reader_tensors(reader):
        torch_tensor = torch.from_numpy(np.array(tensor.data, copy=False))
        if torch_dtype is not None:
            torch_tensor = torch_tensor.to(torch_dtype)
        torch_tensor = torch_tensor.to(device=device)
        state_dict[tensor.name] = torch_tensor

    return state_dict


__all__ = ["GGUFImportError", "load_gguf_state_dict"]

