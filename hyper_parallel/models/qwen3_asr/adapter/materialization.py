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
"""Rebuild Qwen3-ASR's deterministic non-persistent buffers after ``to_empty()``."""

from typing import Any

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.models.materialization import (
    MaterializationContext,
    register_rebuildable_buffer,
    register_rope_buffers,
)

_POSITIONAL_EMBEDDING = "positional_embedding"


def _build_positional_table(module: Any) -> torch.Tensor:
    """Recompute the sinusoidal table the module would have built in ``__init__``."""
    compute = getattr(module, "compute_default_singular_positional_embedding", None)
    if not callable(compute):
        raise TypeError(
            f"{type(module).__name__} cannot rebuild {_POSITIONAL_EMBEDDING!r}: "
            "the sinusoidal generator is missing"
        )
    return compute()


def _register_audio_positional_table(model: Any) -> None:
    """Declare the speech tower's sinusoidal table as rebuildable."""
    audio_tower = getattr(getattr(model, "model", None), "audio_tower", None)
    embedding = getattr(audio_tower, _POSITIONAL_EMBEDDING, None)
    if embedding is None:
        raise ValueError(
            "Qwen3-ASR materialization expected model.model.audio_tower.positional_embedding"
        )

    def factory(_: MaterializationContext) -> torch.Tensor:
        return _build_positional_table(embedding)

    register_rebuildable_buffer(embedding, _POSITIONAL_EMBEDDING, factory=factory)


def _register_text_rope(model: Any) -> None:
    """Declare the Qwen3 text tower's RoPE frequencies as rebuildable."""
    language_model = getattr(getattr(model, "model", None), "language_model", None)
    rotary = getattr(language_model, "rotary_emb", None)
    if rotary is None:
        raise ValueError(
            "Qwen3-ASR materialization expected model.model.language_model.rotary_emb"
        )

    register_rope_buffers(rotary)


def build_qwen3_asr_materialization(model: Any) -> None:
    """Register every non-persistent derived buffer as rebuildable.

    The speech tower's sinusoidal table and the text tower's RoPE frequencies
    are non-persistent buffers: absent from the checkpoint and left
    uninitialized by ``to_empty()``. Both are deterministic from the module
    geometry and config, so they are rebuilt rather than promoted to
    checkpoint state.

    Args:
        model: Constructed Qwen3-ASR model.
    """
    _register_audio_positional_table(model)
    _register_text_rope(model)


__all__ = ["build_qwen3_asr_materialization"]
