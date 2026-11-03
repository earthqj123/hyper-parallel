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
"""Qwen3-ASR adapter contracts: derived state rebuild and sample token padding."""

import unittest
from types import SimpleNamespace

import torch

from hyper_parallel.models.materialization import (
    MaterializationContext,
    rebuild_materialized_state,
)
from hyper_parallel.models.qwen3_asr.adapter.materialization import (
    build_qwen3_asr_materialization,
)
from tests.common.mark_utils import arg_mark


class _SinusoidsPositionEmbedding(torch.nn.Module):
    """Mirror the released module's non-persistent sinusoidal table."""

    def __init__(self, length: int = 13, channels: int = 8) -> None:
        super().__init__()
        self.length = length
        self.channels = channels
        self.register_buffer(
            "positional_embedding",
            self.compute_default_singular_positional_embedding(),
            persistent=False,
        )

    def compute_default_singular_positional_embedding(self) -> torch.Tensor:
        """Return the deterministic table used by the released implementation."""
        positions = torch.arange(self.length, dtype=torch.float32).unsqueeze(1)
        half = torch.arange(0, self.channels, 2, dtype=torch.float32) / self.channels
        inverse = torch.exp(half * -(torch.log(torch.tensor(10000.0))))
        table = torch.zeros(self.length, self.channels, dtype=torch.float32)
        table[:, 0::2] = torch.sin(positions * inverse)
        table[:, 1::2] = torch.cos(positions * inverse)
        return table


class _RotaryEmbedding(torch.nn.Module):
    """Mirror ``Qwen3RotaryEmbedding``'s non-persistent frequency buffers."""

    rope_type = "default"

    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.config = SimpleNamespace(rope_parameters={"rope_type": "default", "rope_theta": 10000.0})
        self.config.head_dim = dim
        inv_freq = self.compute_default_rope_parameters(self.config, device=None)[0]
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

    @staticmethod
    def compute_default_rope_parameters(config, device=None, seq_len=None):
        """Copy the released default-RoPE frequency computation."""
        del seq_len
        base = config.rope_parameters["rope_theta"]
        dim = config.head_dim
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, 1.0


class _AsrLikeModel(torch.nn.Module):
    """Hold the two towers' derived state under the released attribute paths."""

    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        audio_tower = torch.nn.Module()
        audio_tower.positional_embedding = _SinusoidsPositionEmbedding()
        self.model.audio_tower = audio_tower
        self.model.language_model = torch.nn.Module()
        self.model.language_model.rotary_emb = _RotaryEmbedding()


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

        build_qwen3_asr_materialization(model)
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
        Expectation: The missing ``rotary_emb`` is reported instead of silently
        leaving the RoPE buffers uninitialized.
        """
        model = _AsrLikeModel()
        del model.model.language_model.rotary_emb

        with self.assertRaises(ValueError):
            build_qwen3_asr_materialization(model)


def _fake_processor(pad_token_id: int = 0, eos_token: str = "<|im_end|>"):
    """Build the minimal processor surface the ASR transform touches."""
    return SimpleNamespace(
        tokenizer=SimpleNamespace(pad_token_id=pad_token_id, eos_token=eos_token),
    )


class _StubAsrTransform:
    """Exercise token padding without decoding audio."""

    def __init__(self, pad_to_length, pad_token_id=0):
        from hyper_parallel.models.qwen3_asr.adapter.data.transform_fn import Qwen3ASRTransform

        self._transform = Qwen3ASRTransform.__new__(Qwen3ASRTransform)
        self._transform.processor = _fake_processor(pad_token_id)
        self._transform.audio_pad_frames = None
        self._transform.pad_to_length = pad_to_length

    def pad(self, length):
        input_ids = torch.arange(1, length + 1)
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        labels[:2] = -100
        return self._transform._pad_tokens(input_ids, attention_mask, labels)


class TestQwen3AsrTokenPadding(unittest.TestCase):
    """Fixed-length token padding makes variable-length utterances stackable."""

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


if __name__ == "__main__":
    unittest.main()
