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
"""Register the Qwen3-ASR architecture and its adapter providers."""

from hyper_parallel.models.adapter_spec import ModelAdapterSpec
from hyper_parallel.models.qwen3_asr.adapter.materialization import (
    build_qwen3_asr_materialization,
)
from hyper_parallel.models.qwen3_asr.adapter.policies.sharding import (
    build_parameter_sharding_rules,
)
from hyper_parallel.models.registry import register_custom_model, register_model_adapter

# The released architecture is only reachable through
# ``AutoModelForSpeechSeq2Seq`` in Transformers, which the HyperParallel
# builders never consult. Registering it as a custom model keeps construction
# on the concrete class instead of failing to resolve it from ``AutoModel``.
# The family re-exports the released class with the ``from_config`` entry the
# config-first build path requires (removed from PreTrainedModel in
# Transformers 5.13).
register_custom_model(
    "Qwen3ASRForConditionalGeneration",
    "hyper_parallel.models.qwen3_asr.modeling_qwen3_asr",
    "Qwen3ASRForCausalLM",
)

QWEN3_ASR_ADAPTER_SPEC = ModelAdapterSpec(
    architecture="Qwen3ASRForConditionalGeneration",
    model_type="qwen3_asr",
    materialization=build_qwen3_asr_materialization,
    sharding_rules=build_parameter_sharding_rules,
)

register_model_adapter(QWEN3_ASR_ADAPTER_SPEC)


__all__ = ["QWEN3_ASR_ADAPTER_SPEC"]
