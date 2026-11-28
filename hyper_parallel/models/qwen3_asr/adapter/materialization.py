# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Rebuild Qwen3-ASR's deterministic non-persistent buffers after ``to_empty()``.

Two branches hold state that never enters a checkpoint:

* the speech tower's sinusoidal ``positional_embedding`` table, whose values are
  computed from the module geometry in ``__init__``;
* the Qwen3 text tower's RoPE frequencies (``inv_freq`` /
  ``original_inv_freq``), computed from the effective config.

Meta construction (``init_device: meta``) leaves those buffers on the meta
device, which the generic materializer cannot preserve across ``to_empty()``.
Both are deterministic, so the adapter registers a CPU reference value that
``rebuild_materialized_state`` copies onto the materialized module.
"""

from __future__ import annotations

import torch
from torch import nn

from hyper_parallel.models.materialization import register_rebuildable_buffer

_POSITIONAL_EMBEDDING = "positional_embedding"
_ROPE_BUFFERS = ("inv_freq", "original_inv_freq")


def _is_sinusoidal_embedding(module: nn.Module) -> bool:
    """Return whether the module owns a regenerable sinusoidal position table."""
    return (
        callable(getattr(module, "compute_default_singular_positional_embedding", None))
        and _POSITIONAL_EMBEDDING in module._buffers  # pylint: disable=protected-access
    )


def _rope_buffer_names(module: nn.Module) -> tuple[str, ...]:
    """Return the RoPE frequency buffers one module owns, if it is a RoPE module."""
    if not hasattr(module, "config"):
        return ()
    return tuple(
        name for name in _ROPE_BUFFERS
        if name in module._buffers  # pylint: disable=protected-access
    )


def _register_positional_table(module: nn.Module) -> None:
    """Recompute the sinusoids table from the module geometry on CPU."""
    # Computed on CPU: NPU transcendental kernels differ by FP32 ULPs, which
    # must not decide the table the released ``__init__`` would have produced.
    with torch.device("cpu"):
        table = module.compute_default_singular_positional_embedding()
    value = table.to(dtype=getattr(module, _POSITIONAL_EMBEDDING).dtype)
    register_rebuildable_buffer(module, _POSITIONAL_EMBEDDING, value=value, persistent=False)


def _register_rope_frequencies(module: nn.Module, names: tuple[str, ...]) -> None:
    """Rebuild RoPE frequencies from a CPU reference instance of the module class."""
    with torch.device("cpu"):
        reference = type(module)(module.config, device="cpu")
    for name in names:
        value = getattr(reference, name).to(dtype=getattr(module, name).dtype)
        register_rebuildable_buffer(module, name, value=value, persistent=False)


def register_derived_buffers(model: nn.Module) -> None:
    """Declare every non-persistent derived buffer of a Qwen3-ASR model.

    Args:
        model: Constructed Qwen3-ASR model (meta or real storage).

    Raises:
        ValueError: If either tower's derived state is missing, which means the
            model layout no longer matches this adapter.
    """
    sinusoidal = 0
    rotary = 0
    for module in model.modules():
        if _is_sinusoidal_embedding(module):
            _register_positional_table(module)
            sinusoidal += 1
            continue
        names = _rope_buffer_names(module)
        if names:
            _register_rope_frequencies(module, names)
            rotary += 1

    if not sinusoidal:
        raise ValueError("Qwen3-ASR materialization found no sinusoidal positional table")
    if not rotary:
        raise ValueError("Qwen3-ASR materialization found no rotary frequency buffers")


__all__ = ["register_derived_buffers"]
