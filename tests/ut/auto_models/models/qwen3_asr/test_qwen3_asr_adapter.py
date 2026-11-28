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
"""Qwen3-ASR adapter contracts: derived state rebuild and sample encoding."""

import math
import unittest
from types import SimpleNamespace

import torch

from hyper_parallel.models.materialization import (
    MaterializationContext,
    rebuild_materialized_state,
)
from hyper_parallel.models.qwen3_asr.adapter.data.transform_fn import (
    Qwen3ASRTransform,
    _as_messages,
)
from hyper_parallel.models.qwen3_asr.adapter.materialization import (
    register_derived_buffers,
)
from tests.common.mark_utils import arg_mark


class _SinusoidsPositionEmbedding(torch.nn.Module):
    """Mirror the released module's non-persistent sinusoidal table."""

    def __init__(self, length: int = 13, channels: int = 8, max_timescale: float = 10000) -> None:
        super().__init__()
        self.length = length
        self.channels = channels
        self.max_timescale = max_timescale
        self.register_buffer(
            "positional_embedding",
            self.compute_default_singular_positional_embedding(),
            persistent=False,
        )

    def compute_default_singular_positional_embedding(self) -> torch.Tensor:
        """Return the deterministic table the released implementation builds."""
        log_timescale_increment = math.log(self.max_timescale) / (self.channels // 2 - 1)
        inv_timescales = torch.exp(-log_timescale_increment * torch.arange(self.channels // 2).float())
        scaled_time = torch.arange(self.length)[:, None] * inv_timescales[None, :]
        return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class _RotaryEmbedding(torch.nn.Module):
    """Mirror the text tower's non-persistent RoPE frequency buffers."""

    def __init__(self, config: SimpleNamespace, device: str | torch.device | None = None) -> None:
        super().__init__()
        self.config = config
        dim = config.head_dim
        inv_freq = 1.0 / (
            config.rope_theta ** (
                torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)


def _text_config() -> SimpleNamespace:
    """Return the subset of the text config the reference RoPE module reads."""
    return SimpleNamespace(head_dim=8, rope_theta=10000.0)


class _AsrLikeModel(torch.nn.Module):
    """Hold the two towers' derived state under the released attribute paths."""

    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        audio_tower = torch.nn.Module()
        audio_tower.positional_embedding = _SinusoidsPositionEmbedding()
        self.model.audio_tower = audio_tower
        self.model.language_model = torch.nn.Module()
        self.model.language_model.rotary_emb = _RotaryEmbedding(_text_config())


class TestQwen3AsrMaterialization(unittest.TestCase):
    """Derived buffers survive a ``to_empty``-style storage wipe."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_rope_and_positional_tables_are_restored(self):
        """Feature: derived-buffer rebuild after materialization.

        Description: Wipe every buffer value the way ``to_empty()`` does, then
        run the adapter's rebuild lifecycle.
        Expectation: The sinusoidal table and both RoPE frequencies match the
        values the modules computed in ``__init__``.
        """
        model = _AsrLikeModel()
        rotary = model.model.language_model.rotary_emb
        embedding = model.model.audio_tower.positional_embedding
        expected_inv = rotary.inv_freq.clone()
        expected_original_inv = rotary.original_inv_freq.clone()
        expected_positional = embedding.positional_embedding.clone()

        register_derived_buffers(model)
        with torch.no_grad():
            for buffer in rotary._buffers.values():
                buffer.zero_()
            embedding.positional_embedding.zero_()

        rebuild_materialized_state(
            model,
            MaterializationContext(reason="checkpoint_load", device=torch.device("cpu"), strict=False),
        )

        torch.testing.assert_close(rotary.inv_freq, expected_inv)
        torch.testing.assert_close(rotary.original_inv_freq, expected_original_inv)
        torch.testing.assert_close(embedding.positional_embedding, expected_positional)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_materialization_requires_both_towers(self):
        """Feature: adapter fails loudly on an unexpected model layout.

        Description: Call the provider on a model without the text tower.
        Expectation: The missing RoPE buffers are reported instead of silently
        leaving them uninitialized.
        """
        model = _AsrLikeModel()
        del model.model.language_model.rotary_emb

        with self.assertRaises(ValueError):
            register_derived_buffers(model)


def _fake_processor(pad_token_id: int = 0, eos_token: str = "<|im_end|>"):
    """Build the minimal processor surface the ASR transform touches."""
    return SimpleNamespace(
        tokenizer=SimpleNamespace(pad_token_id=pad_token_id, eos_token=eos_token),
    )


class _StubAsrTransform:
    """Exercise token and mel padding without decoding audio."""

    def __init__(self, pad_to_length=None, audio_pad_frames=None, pad_token_id=0):
        self._transform = Qwen3ASRTransform.__new__(Qwen3ASRTransform)
        self._transform.processor = _fake_processor(pad_token_id)
        self._transform.audio_pad_frames = audio_pad_frames
        self._transform.pad_to_length = pad_to_length

    def pad(self, length):
        input_ids = torch.arange(1, length + 1)
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        labels[:2] = -100
        return self._transform._pad_tokens(input_ids, attention_mask, labels)

    def pad_audio(self, frames):
        features = torch.ones(4, frames)
        mask = torch.ones(frames, dtype=torch.long)
        return self._transform._pad_audio(features, mask)


class TestQwen3AsrTokenPadding(unittest.TestCase):
    """Fixed-length padding makes variable-length utterances stackable."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_uneven_utterances_stack_to_one_tensor(self):
        """Feature: multi-sample micro-batches.

        Description: Pad two different-length utterances to one fixed length.
        Expectation: They stack, padding uses the tokenizer pad id, and padded
        label positions stay ignored.
        """
        stub = _StubAsrTransform(pad_to_length=16, pad_token_id=7)
        short = stub.pad(5)
        long = stub.pad(11)

        batch = torch.stack([short[0], long[0]])
        self.assertEqual(tuple(batch.shape), (2, 16))
        self.assertEqual(int(short[0][-1]), 7)
        torch.testing.assert_close(short[1][5:], torch.zeros(11, dtype=short[1].dtype))
        torch.testing.assert_close(short[2][5:], torch.full((11,), -100))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_padding_is_disabled_by_default(self):
        """Feature: per-sample lengths stay untouched when padding is unset.

        Description: Pad with ``pad_to_length=None``.
        Expectation: The utterance keeps its own length.
        """
        stub = _StubAsrTransform(pad_to_length=None)
        input_ids, attention_mask, labels = stub.pad(5)

        self.assertEqual(tuple(input_ids.shape), (5,))
        self.assertEqual(tuple(attention_mask.shape), (5,))
        self.assertEqual(tuple(labels.shape), (5,))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_long_utterance_is_truncated_to_the_fixed_length(self):
        """Feature: a sample above the fixed length stays batchable.

        Description: Pad a 20-token utterance with ``pad_to_length=16``.
        Expectation: The tail is dropped and the shape is exactly 16.
        """
        stub = _StubAsrTransform(pad_to_length=16, pad_token_id=7)
        input_ids, attention_mask, labels = stub.pad(20)

        self.assertEqual(tuple(input_ids.shape), (16,))
        self.assertEqual(int(input_ids[-1]), 16)
        self.assertEqual(int(attention_mask[-1]), 1)
        self.assertEqual(int(labels[-1]), 16)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_mel_tensor_pads_to_the_configured_frame_count(self):
        """Feature: mel stacking for multi-sample micro-batches.

        Description: Pad a 7-frame spectrogram with ``audio_pad_frames=10``.
        Expectation: Values pad with zeros and the frame mask pads with zeros.
        """
        features, mask = _StubAsrTransform(audio_pad_frames=10).pad_audio(7)

        self.assertEqual(tuple(features.shape), (4, 10))
        self.assertEqual(tuple(mask.shape), (10,))
        torch.testing.assert_close(features[:, 7:], torch.zeros(4, 3))
        torch.testing.assert_close(mask[7:], torch.zeros(3, dtype=mask.dtype))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_audio_longer_than_the_budget_is_rejected(self):
        """Feature: an over-long clip cannot be silently cropped.

        Description: Pad a 12-frame spectrogram with ``audio_pad_frames=10``.
        Expectation: The transform raises instead of dropping frames.
        """
        with self.assertRaises(ValueError):
            _StubAsrTransform(audio_pad_frames=10).pad_audio(12)


class TestQwen3AsrRecordLayouts(unittest.TestCase):
    """Both supported record layouts normalize to one chat conversation."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_released_row_layout_is_normalized(self):
        """Feature: the released Qwen3-ASR fine-tuning row.

        Description: Normalize ``{"audio", "text"}``.
        Expectation: A user audio turn followed by the assistant transcript.
        """
        messages = _as_messages({
            "audio": "wavs/utt0000.wav",
            "text": "language English<asr_text>hello",
        })

        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[0]["content"], [{"type": "audio", "audio": "wavs/utt0000.wav"}])
        self.assertEqual(messages[1], {"role": "assistant", "content": "language English<asr_text>hello"})

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_chat_layout_is_passed_through(self):
        """Feature: the Omni chat record.

        Description: Normalize a record that already carries ``messages``.
        Expectation: The conversation is returned unchanged.
        """
        messages = [
            {"role": "user", "content": [{"type": "audio", "audio": "wavs/u.wav"}]},
            {"role": "assistant", "content": "hi"},
        ]

        self.assertIs(_as_messages({"messages": messages}), messages)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_rows_without_audio_or_transcript_are_filtered(self):
        """Feature: source filtering.

        Description: Validate rows missing the audio or the transcript.
        Expectation: ``is_valid_sample`` rejects them and normalization raises.
        """
        transform = Qwen3ASRTransform.__new__(Qwen3ASRTransform)

        self.assertFalse(transform.is_valid_sample({"audio": "", "text": "x"}))
        self.assertFalse(transform.is_valid_sample({"audio": "a.wav", "text": "  "}))
        self.assertFalse(transform.is_valid_sample({"unrelated": 1}))
        self.assertTrue(transform.is_valid_sample({"audio": "a.wav", "text": "hello"}))
        with self.assertRaises(ValueError):
            _as_messages({"unrelated": 1})


if __name__ == "__main__":
    unittest.main()
