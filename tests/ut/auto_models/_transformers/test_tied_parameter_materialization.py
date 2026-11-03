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
"""Keep declared tied parameters sharing storage across device materialization."""
import unittest

import torch
from torch import nn

from hyper_parallel.models._transformers.model_builder import (
    _move_model_to_device,
    _restore_tied_parameters,
)
from tests.common.mark_utils import arg_mark


class _TiedStack(nn.Module):
    """Minimal module declaring an embed/lm_head tie in Transformers' form."""

    def __init__(self, vocab_size: int = 16, hidden_size: int = 8, with_tie: bool = True) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.all_tied_weights_keys = {"head.weight": "embed.weight"} if with_tie else {}
        if with_tie:
            self.head.weight = self.embed.weight

    def tie_weights(self, recompute_mapping: bool = True) -> None:
        """Follow PreTrainedModel's contract: bind the target to the source."""
        del recompute_mapping
        self.head.weight = self.embed.weight


def _shares_storage(module: _TiedStack) -> bool:
    """Report whether both ends point at the same physical storage."""
    embed = module.embed.weight
    head = module.head.weight
    return embed is head and (
        embed.untyped_storage().data_ptr() == head.untyped_storage().data_ptr()
    )


class TestTiedParameterMaterialization(unittest.TestCase):
    """Verify the tie survives the meta-to-device move used by pretrained loading."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_to_empty_keeps_declared_tie(self):
        """Feature: tied-parameter initialization after materialization.

        Description: Materialize a meta-constructed model that declares a tie.
        Expectation: Both ends still share one storage after ``to_empty``.
        """
        model = _TiedStack().to(device="meta")
        self.assertTrue(model.embed.weight.is_meta)

        model = _move_model_to_device(model, is_meta_device=True, device=torch.device("cpu"))

        self.assertTrue(_shares_storage(model))

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_restore_leaves_untied_models_alone(self):
        """Feature: tied-parameter initialization is opt-in per model.

        Description: Model without a tie declaration goes through the same restore hook.
        Expectation: The two parameters stay independent.
        """
        model = _TiedStack(with_tie=False)
        original_head = model.head.weight

        _restore_tied_parameters(model)

        self.assertIs(model.head.weight, original_head)
        self.assertIsNot(model.head.weight, model.embed.weight)


if __name__ == "__main__":
    unittest.main()
