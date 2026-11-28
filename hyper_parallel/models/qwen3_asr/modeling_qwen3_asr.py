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
"""Qwen3-ASR training model: the released architecture plus a config builder.

Transformers 5.13 removed the public ``PreTrainedModel.from_config`` and kept
only the private ``_from_config`` (which applies dtype and
``attn_implementation`` to the config before construction). The HyperParallel
custom-model path constructs from config and deferred-loads the checkpoint
afterwards, so the family re-exports the released class with the entry point
it needs. No module, parameter, or forward behaviour is changed.
"""

from typing import Any

from transformers.models.qwen3_asr.modeling_qwen3_asr import (
    Qwen3ASRCausalLMOutputWithPast,
    Qwen3ASRForConditionalGeneration,
)


class Qwen3ASRForCausalLM(Qwen3ASRForConditionalGeneration):
    """Released Qwen3-ASR conditional-generation model with ``from_config``."""

    @classmethod
    def from_config(cls, config: Any, **kwargs: Any) -> "Qwen3ASRForCausalLM":
        """Construct the model from configuration without loading a checkpoint.

        Args:
            config: ``Qwen3ASRConfig`` resolved from the checkpoint directory.
            **kwargs: Model construction options; dtype and
                ``attn_implementation`` are applied by ``_from_config``.

        Returns:
            An uninitialized (or randomly initialized) model instance.
        """
        return cls._from_config(config, **kwargs)  # pylint: disable=protected-access


__all__ = [
    "Qwen3ASRForCausalLM",
    "Qwen3ASRForConditionalGeneration",
    "Qwen3ASRCausalLMOutputWithPast",
]
