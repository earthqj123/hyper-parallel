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
"""Generate a small synthetic ASR dataset in the Qwen3-ASR JSONL format.

The emitted records match the released fine-tuning contract: one utterance per
line with ``audio`` (a WAV path relative to the JSONL file) and ``text`` (the
transcript, carrying the optional ``language <NAME><asr_text>`` prefix). Audio
is a synthetic tone/ noise mixture -- this script exists to exercise the
training pipeline, not to produce a usable speech corpus.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

SAMPLING_RATE = 16000

_TRANSCRIPTS = (
    "the quick brown fox jumps over the lazy dog",
    "speech recognition maps audio to text",
    "this sample validates the training pipeline",
    "hyper parallel runs the encoder and decoder",
    "short utterance",
    "a considerably longer transcription that exercises the language model head",
)


def _synthesize(duration_seconds: float, seed: int) -> np.ndarray:
    """Build a deterministic waveform with a varying fundamental frequency."""
    rng = np.random.default_rng(seed)
    sample_count = int(SAMPLING_RATE * duration_seconds)
    times = np.arange(sample_count, dtype=np.float32) / SAMPLING_RATE
    frequency = 120.0 + 40.0 * (seed % 5)
    tone = 0.15 * np.sin(2 * math.pi * frequency * times)
    harmonic = 0.05 * np.sin(2 * math.pi * 2 * frequency * times)
    noise = 0.01 * rng.standard_normal(sample_count).astype(np.float32)
    envelope = np.minimum(1.0, np.minimum(times * 20.0, (duration_seconds - times) * 20.0))
    return ((tone + harmonic + noise) * envelope).astype(np.float32)


def main() -> None:
    """Write ``wavs/`` and an accompanying ``train.jsonl``."""
    parser = argparse.ArgumentParser(description="Generate a synthetic Qwen3-ASR SFT dataset")
    parser.add_argument("--output-dir", required=True, help="Directory receiving wavs/ and train.jsonl")
    parser.add_argument("--num-samples", type=int, default=8, help="Number of utterances to generate")
    parser.add_argument("--language", default="English", help="Language name used in the transcript prefix")
    args = parser.parse_args()

    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")

    import soundfile as sf  # pylint: disable=import-outside-toplevel

    output_dir = Path(args.output_dir)
    wav_dir = output_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for index in range(args.num_samples):
        duration = 1.5 + 0.5 * (index % 4)
        waveform = _synthesize(duration, seed=index)
        wav_path = wav_dir / f"utt{index:04d}.wav"
        sf.write(wav_path, waveform, SAMPLING_RATE)

        transcript = _TRANSCRIPTS[index % len(_TRANSCRIPTS)]
        records.append({
            "audio": f"wavs/{wav_path.name}",
            "text": f"language {args.language}<asr_text>{transcript}",
        })

    manifest = output_dir / "train.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} utterances to {wav_dir}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
