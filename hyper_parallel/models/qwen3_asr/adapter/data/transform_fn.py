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
"""Qwen3-ASR sample encoding, backed by the model's native processor."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.data.constants import IGNORE_INDEX, ONLINE_SOURCE_PATH_KEY
from hyper_parallel.data.dataset_logging import get_dataset_logger
from hyper_parallel.data.omni.omni_transform import OmniDataTransform, build_labels_from_supervised_start

logger = get_dataset_logger("hyper_parallel.data.qwen3_asr")

SAMPLING_RATE = 16000


def _load_audio(path: str) -> np.ndarray:
    """Read one audio file as mono float32 at the model's sampling rate.

    Args:
        path: Local audio path.

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


class Qwen3ASRTransform(OmniDataTransform):
    """Encode ``{"audio": path, "text": transcript}`` rows for ASR SFT.

    The record contract follows the released Qwen3-ASR fine-tuning format: one
    JSONL row per utterance, with the transcript carrying the optional
    ``language <NAME><asr_text>`` prefix. Supervision starts at the first
    transcript token, so the prompt (system header, audio placeholders and the
    assistant header) carries no gradient.
    """

    def __init__(
            self,
            processor: Any,
            *,
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
                fixed value when a micro-batch holds more than one sample and
                their mel tensors must stack.
            pad_to_length: Pad every token field to this exact length. The Omni
                collator stacks samples before any alignment padding, so a
                micro-batch of more than one utterance needs a fixed token
                length. Leave ``None`` for per-sample lengths.
        """
        super().__init__(max_seq_len=max_seq_len, processor=processor, training_mode=training_mode)
        if audio_pad_frames is not None and audio_pad_frames <= 0:
            raise ValueError("audio_pad_frames must be a positive integer or None")
        if pad_to_length is not None and not 0 < pad_to_length <= max_seq_len:
            raise ValueError("pad_to_length must be a positive integer no larger than max_seq_len")
        self.audio_pad_frames = audio_pad_frames
        self.pad_to_length = pad_to_length

    def is_valid_sample(self, sample: Mapping[str, Any]) -> bool:
        """Keep rows that carry both an audio reference and a transcript."""
        audio = sample.get("audio")
        text = sample.get("text")
        return isinstance(audio, str) and bool(audio) and isinstance(text, str) and bool(text.strip())

    def _resolve_audio_path(self, sample: Mapping[str, Any]) -> str:
        """Anchor a relative audio reference to the record's source file."""
        audio_path = str(sample["audio"])
        source_path = sample.get(ONLINE_SOURCE_PATH_KEY)
        if not source_path:
            return os.path.normpath(audio_path)
        return self._resolve_media_path(audio_path, os.path.dirname(str(source_path)))

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

    def _encode_prompt(self, prompt_text: str, waveform: np.ndarray) -> dict[str, torch.Tensor]:
        """Run the native processor on the rendered prompt.

        Both the mel features and the audio placeholders depend only on the
        waveform and the prompt, so encoding the prompt once and appending the
        tokenized transcript avoids a second full feature-extraction pass.
        """
        return self.processor(
            text=[prompt_text],
            audio=[waveform],
            return_tensors="pt",
            padding=False,
        )

    def encode_sample(self, sample: dict[str, Any]) -> dict[str, Any] | None:
        """Encode one utterance into model inputs with prompt-only masked labels."""
        if self.training_mode != "sft":
            raise NotImplementedError("Qwen3-ASR pretraining requires a model-specific encoding method")

        audio_path = self._resolve_audio_path(sample)
        waveform = _load_audio(audio_path)
        transcript = str(sample["text"]).strip()

        prompt_text = self.processor.apply_chat_template(
            [[{"role": "user", "content": [{"type": "audio", "audio": waveform}]}]],
            tokenize=False,
            add_generation_prompt=True,
        )[0]
        eos = self.processor.tokenizer.eos_token or ""

        prompt = self._encode_prompt(prompt_text, waveform)
        prompt_ids = prompt["input_ids"].squeeze(0)
        supervised_start = int(prompt_ids.numel())
        # add_special_tokens=False keeps the transcript boundary exact: the
        # template already emitted the assistant header.
        transcript_ids = self.processor.tokenizer(
            transcript + eos, add_special_tokens=False,
        )["input_ids"]
        if not transcript_ids:
            logger.warning("Skipping %s: empty transcript after tokenization", audio_path)
            return None

        input_ids = torch.cat([prompt_ids, torch.tensor(transcript_ids, dtype=prompt_ids.dtype)])

        if input_ids.numel() > self.max_seq_len:
            if supervised_start >= self.max_seq_len:
                logger.warning("Skipping %s: prompt exceeds max_seq_len", audio_path)
                return None
            input_ids = input_ids[: self.max_seq_len]

        features, features_mask = self._pad_audio(
            prompt["input_features"][0], prompt["input_features_mask"][0],
        )
        labels = build_labels_from_supervised_start(input_ids, supervised_start)
        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = IGNORE_INDEX

        attention_mask = torch.ones_like(input_ids)
        input_ids, attention_mask, labels = self._pad_tokens(input_ids, attention_mask, labels)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "input_features": features,
            "input_features_mask": features_mask,
        }

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
        processor,
        max_seq_len=max_seq_len,
        training_mode=training_mode,
        audio_pad_frames=audio_pad_frames,
        pad_to_length=pad_to_length,
    )


__all__ = ["Qwen3ASRTransform", "build_qwen3_asr_transform"]
