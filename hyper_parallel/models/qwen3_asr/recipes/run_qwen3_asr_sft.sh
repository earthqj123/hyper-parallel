#!/bin/bash
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

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../../../.." && pwd)
OUTPUT_DIR="${PROJECT_ROOT}/output/training_demo/qwen3_asr"
PLOG_DIR="${OUTPUT_DIR}/plog"

RUN_NAME=${RUN_NAME:-qwen3_asr_sft}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}

if [[ ! ${RUN_NAME} =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "RUN_NAME may contain only letters, digits, dots, underscores, and hyphens" >&2
    exit 1
fi
if [[ ! ${NPROC_PER_NODE} =~ ^[1-9][0-9]*$ ]]; then
    echo "NPROC_PER_NODE must be a positive integer" >&2
    exit 1
fi

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <model_dir> <data_jsonl> [extra --key=value ...]" >&2
    exit 1
fi

MODEL_PATH=$(cd "$1" 2>/dev/null && pwd) || {
    echo "Model directory does not exist: $1" >&2
    exit 1
}
shift
DATA_PATH=$(cd "$(dirname "$1")" 2>/dev/null && pwd)/$(basename "$1") || {
    echo "Data file does not exist: $1" >&2
    exit 1
}
shift

if [[ ! -s "${MODEL_PATH}/config.json" ]]; then
    echo "Model directory is missing config.json: ${MODEL_PATH}" >&2
    exit 1
fi
if [[ ! -s "${DATA_PATH}" ]]; then
    echo "Training data is missing or empty: ${DATA_PATH}" >&2
    exit 1
fi

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}" "${PLOG_DIR}"

export ASCEND_PROCESS_LOG_PATH="${PLOG_DIR}"
export ASCEND_GLOBAL_LOG_LEVEL=1
export ASCEND_SLOG_PRINT_TO_STDOUT=0
# 同步下发会串行化所有 NPU kernel，默认关闭；排查精度/设备问题时显式置 1。
export ASCEND_LAUNCH_BLOCKING=${ASCEND_LAUNCH_BLOCKING:-0}
export PYTHONFAULTHANDLER=1
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

# Bare-metal CANN images ship torchrun; environments installed from the wheel
# alone expose the equivalent entry point only as a module.
if command -v torchrun >/dev/null 2>&1; then
    LAUNCHER=(torchrun)
else
    LAUNCHER=("${PYTHON_BIN:-python}" -m torch.distributed.run)
fi

"${LAUNCHER[@]}" \
    --standalone \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --module scripts.train_vl \
    "${SCRIPT_DIR}/train.yaml" \
    --model.pretrained_model_name_or_path="${MODEL_PATH}" \
    --dataset.model_assets.pretrained_model_name_or_path="${MODEL_PATH}" \
    --dataset.data_path="${DATA_PATH}" \
    --training.global_batch_size="${NPROC_PER_NODE}" \
    --fsdp_config.dp_shard_size="${NPROC_PER_NODE}" \
    "$@" \
    2>&1 | tee "${OUTPUT_DIR}/run_${RUN_NAME}.log"
