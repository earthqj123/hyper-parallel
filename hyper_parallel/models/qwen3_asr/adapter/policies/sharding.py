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
"""Qwen3-ASR tensor-parallel and FSDP placement policies."""

from typing import Any


def build_parameter_sharding_rules() -> list[tuple[Any, Any]]:
    """Return TP roles for the speech tower and the audio projector.

    The released ``Qwen3ASRAudioAttention`` reshapes ``q_proj`` output to
    ``(seq, num_heads, head_dim)`` over the *full* ``d_model`` and the encoder
    runs over a packed ``cu_seqlens`` layout, so neither the projections nor
    the conv stack can be split along a head axis without rewriting the
    forward.  The projector is a pointwise MLP between two towers and is
    likewise not TP-aware.  Both branches therefore stay replicated; only the
    Qwen3 text tower uses the generic colwise/rowwise rules.
    """
    from hyper_parallel.distributed.tensor_parallel.param_role import (  # pylint: disable=C0415
        ParamRole,
    )

    return [
        (["model.audio_tower."], ParamRole.REPLICATED),
        (["model.multi_modal_projector."], ParamRole.REPLICATED),
    ]


__all__ = ["build_parameter_sharding_rules"]
