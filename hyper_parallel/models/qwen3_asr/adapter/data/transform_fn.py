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
"""Qwen3-ASR sample encoding on the shared Omni data lifecycle.

One record is one utterance: a single audio clip plus its transcript. The
released ``QwenLM/Qwen3-ASR`` fine-tuning format is accepted as-is
(``{"audio": path, "text": transcript}``) and is normalized into the chat
messages the Omni lifecycle carries; the ``messages`` layout is also accepted
directly, with the audio as an ``audio`` content block and the transcript as the
final assistant message.

The native ``Qwen3ASRProcessor`` renders the prompt (system header, audio
placeholder expansion and assistant header) and extracts the mel features; the
transcript is tokenized separately and appended, so supervision starts exactly
at the first transcript token. Prompt-only positions, the audio placeholder and
the assistant header all carry ``IGNORE_INDEX``.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np
import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.data.constants import IGNORE_INDEX
from hyper_parallel.data.dataset_logging import get_dataset_logger
from hyper_parallel.data.omni.omni_transform import (
    OmniDataTransform,
    build_labels_from_supervised_start,
)

logger = get_dataset_logger("hyper_parallel.data.qwen3_asr")

SAMPLING_RATE = 16000
_AUDIO_BLOCK = "audio"


def _load_audio(path: str) -> np.ndarray:
    """Read one audio file as mono float32 at the model's sampling rate.

    Args:
        path: Local audio path, absolute or already resolved by the base class.

    Returns:
        Waveform samples in ``[-1, 1]``.

    Raises:
        ValueError: If the file cannot be decoded.
    """
    import soundfile as sf  # pylint: disable=import-outside-toplevel

    try:
        waveform, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    except Exception as exc:  # noqa: BLE001 - surface the file that failed
        raise ValueError(f"Failed to decode audio {path!r}: {exc}") from exc

    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    if sample_rate != SAMPLING_RATE:
        from scipy.signal import resample_poly  # pylint: disable=import-outside-toplevel

        divisor = math.gcd(int(sample_rate), SAMPLING_RATE)
        # resample_poly keeps float32 for float32 input, so no extra cast.
        waveform = resample_poly(
            waveform, SAMPLING_RATE // divisor, int(sample_rate) // divisor,
        ).astype(np.float32)
    return waveform


def _content_blocks(messages: list) -> Iterator[Mapping[str, Any]]:
    """Yield every list-style content block of a conversation."""
    for message in messages:
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, Mapping):
                yield block


def _text_content(content: Any) -> str:
    """Flatten a transcript given either as a string or as text content blocks."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "".join(
            block.get("text", "") for block in content if isinstance(block, Mapping)
        ).strip()
    return ""


def _as_messages(sample: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return one record's chat messages in either supported layout.

    Args:
        sample: Raw record from the Omni source.

    Returns:
        Chat messages ending with the assistant transcript.

    Raises:
        ValueError: If the record carries neither layout.
    """
    messages = sample.get("messages")
    if isinstance(messages, list) and messages:
        return messages

    audio = sample.get("audio")
    text = sample.get("text")
    if isinstance(audio, str) and audio and isinstance(text, str) and text.strip():
        return [
            {"role": "user", "content": [{"type": _AUDIO_BLOCK, "audio": audio}]},
            {"role": "assistant", "content": text.strip()},
        ]

    raise ValueError(
        "Qwen3-ASR samples need `messages`, or the released `audio` + `text` row"
    )


def _audio_reference(block: Mapping[str, Any]) -> str:
    """Return the path-bearing field of one audio content block."""
    for field in (_AUDIO_BLOCK, "path", "url"):
        value = block.get(field)
        if isinstance(value, str) and value:
            return value
    raise ValueError(f"Qwen3-ASR audio block carries no reference: {dict(block)!r}")


class Qwen3ASRTransform(OmniDataTransform):
    """Encode one speech-text pair into model inputs with transcript-only labels.

    The record contract follows the released Qwen3-ASR fine-tuning format or the
    generic Omni chat layout; see the module docstring. A sample holding more
    than one audio clip is rejected: the native processor expands audio tokens
    one clip at a time, and ASR SFT supervises one utterance per record.
    """

    def __init__(
            self,
            *,
            processor: Any,
            max_seq_len: int = 4096,
            training_mode: str = "sft",
            audio_pad_frames: int | None = None,
            pad_to_length: int | None = None,
    ) -> None:
        """Bind the native processor to the generic Omni lifecycle.

        Args:
            processor: ``Qwen3ASRProcessor`` loaded from the checkpoint.
            max_seq_len: Maximum retained token length per sample.
            training_mode: Training mode forwarded to the Omni lifecycle.
            audio_pad_frames: Pad every mel spectrogram to this many frames.
                Leave ``None`` to keep the processor's per-sample length; set a
                fixed value when a micro-batch holds more than one utterance and
                their mel tensors must stack.
            pad_to_length: Pad every token field to this exact length. The Omni
                collator stacks samples before any alignment padding, so a
                micro-batch of more than one utterance needs a fixed token
                length. Leave ``None`` for per-sample lengths.
        """
        if processor is None:
            raise ValueError("processor is required for the Qwen3-ASR data transform")
        super().__init__(
            max_seq_len=max_seq_len, processor=processor, training_mode=training_mode,
        )
        if audio_pad_frames is not None and audio_pad_frames <= 0:
            raise ValueError("audio_pad_frames must be a positive integer or None")
        if pad_to_length is not None and not 0 < pad_to_length <= max_seq_len:
            raise ValueError("pad_to_length must be a positive integer no larger than max_seq_len")
        self.audio_pad_frames = audio_pad_frames
        self.pad_to_length = pad_to_length

    def is_valid_sample(self, sample: Mapping[str, Any]) -> bool:
        """Keep records that resolve to a chat conversation."""
        try:
            _as_messages(sample)
        except (TypeError, ValueError):
            return False
        return True

    def encode_sample(self, sample: dict[str, Any]) -> dict[str, Any] | None:
        """Encode one utterance into model inputs with prompt-only masked labels."""
        if self.training_mode != "sft":
            raise NotImplementedError("Qwen3-ASR pretraining requires a model-specific encoding method")

        messages = self._prepare_messages(sample)
        if len(messages) < 2 or messages[-1].get("role") != "assistant":
            raise ValueError("Qwen3-ASR sample must end with the assistant transcript")
        transcript = _text_content(messages[-1].get("content"))
        audio_paths = [
            _audio_reference(block) for block in _content_blocks(messages[:-1])
            if block.get("type") == _AUDIO_BLOCK
        ]
        if len(audio_paths) != 1:
            raise ValueError(
                f"Qwen3-ASR SFT expects exactly one audio clip per sample, got {len(audio_paths)}"
            )
        if not transcript:
            raise ValueError("Qwen3-ASR sample has an empty transcript")

        waveform = _load_audio(audio_paths[0])
        prompt_text = self.processor.apply_chat_template(
            [messages[:-1]], tokenize=False, add_generation_prompt=True,
        )[0]
        prompt = self.processor(
            text=[prompt_text],
            audio=[waveform],
            return_tensors="pt",
            padding=False,
            return_attention_mask=True,
        )
        prompt_ids = prompt["input_ids"].squeeze(0)
        supervised_start = int(prompt_ids.numel())

        eos = self.processor.tokenizer.eos_token or ""
        # add_special_tokens=False keeps the transcript boundary exact: the
        # template already emitted the assistant header.
        transcript_ids = self.processor.tokenizer(
            transcript + eos, add_special_tokens=False,
        )["input_ids"]
        if not transcript_ids:
            logger.warning("Skipping %s: empty transcript after tokenization", audio_paths[0])
            return None

        input_ids = torch.cat(
            [prompt_ids, torch.tensor(transcript_ids, dtype=prompt_ids.dtype)],
        )
        if input_ids.numel() > self.max_seq_len:
            if supervised_start >= self.max_seq_len:
                logger.warning("Skipping %s: audio prompt exceeds max_seq_len", audio_paths[0])
                return None
            input_ids = input_ids[: self.max_seq_len]
            logger.debug(
                "Truncated %s to max_seq_len=%d; transcript tail dropped",
                audio_paths[0], self.max_seq_len,
            )

        labels = build_labels_from_supervised_start(input_ids, supervised_start)
        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = IGNORE_INDEX

        attention_mask = torch.ones_like(input_ids)
        input_ids, attention_mask, labels = self._pad_tokens(
            input_ids, attention_mask, labels,
        )
        features, features_mask = self._pad_audio(
            prompt["input_features"][0], prompt["input_features_mask"][0],
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "input_features": features,
            "input_features_mask": features_mask,
        }

    def _prepare_messages(self, sample: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Resolve media paths against the record source before rendering.

        The base class owns path anchoring and external-media binding, so the
        normalized messages are handed back through a record-shaped mapping.
        """
        record = dict(sample)
        record["messages"] = _as_messages(sample)
        return self.prepare_messages(record)

    def _pad_audio(self, features: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Pad the mel axis to ``audio_pad_frames`` when one is configured."""
        if self.audio_pad_frames is None:
            return features, mask

        current = features.shape[-1]
        if current > self.audio_pad_frames:
            raise ValueError(
                f"Audio needs {current} mel frames, above audio_pad_frames={self.audio_pad_frames}"
            )
        pad = self.audio_pad_frames - current
        if pad:
            features = torch.nn.functional.pad(features, (0, pad), value=0.0)
            mask = torch.nn.functional.pad(mask, (0, pad), value=0)
        return features, mask

    def _pad_tokens(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pad one sample's token fields to ``pad_to_length`` when configured.

        A sample longer than ``pad_to_length`` is truncated to it, the same way
        ``max_seq_len`` truncates: the transcript tail is dropped rather than
        silently misaligned against the audio features.
        """
        if self.pad_to_length is None:
            return input_ids, attention_mask, labels

        if input_ids.numel() > self.pad_to_length:
            input_ids = input_ids[: self.pad_to_length]
            attention_mask = attention_mask[: self.pad_to_length]
            labels = labels[: self.pad_to_length]

        pad = self.pad_to_length - input_ids.numel()
        if not pad:
            return input_ids, attention_mask, labels

        pad_token_id = self.processor.tokenizer.pad_token_id or 0
        input_ids = torch.nn.functional.pad(input_ids, (0, pad), value=pad_token_id)
        attention_mask = torch.nn.functional.pad(attention_mask, (0, pad), value=0)
        labels = torch.nn.functional.pad(labels, (0, pad), value=IGNORE_INDEX)
        return input_ids, attention_mask, labels


def build_qwen3_asr_transform(
        *,
        processor: Any,
        max_seq_len: int = 4096,
        training_mode: str = "sft",
        audio_pad_frames: int | None = None,
        pad_to_length: int | None = None,
        **transform_options: Any,
) -> OmniDataTransform:
    """Build the Qwen3-ASR SFT sample transform.

    Args:
        processor: Native ``Qwen3ASRProcessor``.
        max_seq_len: Maximum retained token length.
        training_mode: ``sft`` or ``pretrain``.
        audio_pad_frames: Optional fixed mel length for multi-sample batches.
        pad_to_length: Optional fixed token length for multi-sample batches.
        **transform_options: Reserved for recipe-level overrides.

    Returns:
        The ASR transform instance.
    """
    del transform_options
    return Qwen3ASRTransform(
        processor=processor,
        max_seq_len=max_seq_len,
        training_mode=training_mode,
        audio_pad_frames=audio_pad_frames,
        pad_to_length=pad_to_length,
    )


__all__ = ["Qwen3ASRTransform", "build_qwen3_asr_transform"]
