"""Free Spoken Digit Dataset loader and log-mel feature extractor.

We expect WAV files at data/recordings/<digit>_<speaker>_<idx>.wav after a clone of
github.com/Jakobovski/free-spoken-digit-dataset (or download via scripts/download.py).

Features: 64-band log-mel spectrograms, 25 ms window, 10 ms hop, sample rate 8 kHz.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

SAMPLE_RATE = 8000
N_MELS = 64
WIN_MS = 25
HOP_MS = 10
WIN_LEN = int(SAMPLE_RATE * WIN_MS / 1000)
HOP_LEN = int(SAMPLE_RATE * HOP_MS / 1000)
N_FFT = 256

# CTC vocabulary: 0..9 (label ids 1..10) + blank (id 0).
DIGIT_TOKENS = [str(d) for d in range(10)]
BLANK_ID = 0
VOCAB_SIZE = len(DIGIT_TOKENS) + 1  # +1 for blank


def label_to_id(label: str) -> int:
    return int(label) + 1


def ids_to_str(ids: list[int]) -> str:
    return "".join(DIGIT_TOKENS[i - 1] for i in ids if i != BLANK_ID)


def greedy_ctc_decode(logits: torch.Tensor) -> list[int]:
    """logits: (T, V). Returns collapsed ids (no blanks, no consecutive duplicates)."""
    preds = logits.argmax(dim=-1).tolist()
    out: list[int] = []
    prev = -1
    for p in preds:
        if p != prev and p != BLANK_ID:
            out.append(p)
        prev = p
    return out


@dataclass
class Sample:
    path: Path
    digit: int
    speaker: str


def discover(root: Path) -> list[Sample]:
    rec_dir = root / "recordings"
    if not rec_dir.exists():
        raise FileNotFoundError(
            f"{rec_dir} not found. Run: python scripts/download.py"
        )
    pat = re.compile(r"^(\d)_([a-z]+)_(\d+)\.wav$")
    out: list[Sample] = []
    for p in sorted(rec_dir.glob("*.wav")):
        m = pat.match(p.name)
        if m is None:
            continue
        out.append(Sample(p, int(m.group(1)), m.group(2)))
    return out


def split_by_index(samples: list[Sample], train_max_idx: int = 39) -> tuple[list[Sample], list[Sample]]:
    """FSDD convention: indices 0..4 are test, 5..49 are train. We use idx<5 as test."""
    train, test = [], []
    pat = re.compile(r"^\d_[a-z]+_(\d+)\.wav$")
    for s in samples:
        m = pat.match(s.path.name)
        idx = int(m.group(1))
        (test if idx < 5 else train).append(s)
    return train, test


_mel_extractor: torchaudio.transforms.MelSpectrogram | None = None


def _get_mel() -> torchaudio.transforms.MelSpectrogram:
    global _mel_extractor
    if _mel_extractor is None:
        _mel_extractor = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=N_FFT, win_length=WIN_LEN, hop_length=HOP_LEN,
            n_mels=N_MELS, power=2.0,
        )
    return _mel_extractor


def load_wav(path: Path) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    return wav.squeeze(0)


def compute_logmel(wav: torch.Tensor) -> torch.Tensor:
    """wav: (T,) at SAMPLE_RATE. Returns (n_mels, frames) log-mel."""
    mel = _get_mel()(wav.unsqueeze(0)).squeeze(0)
    return torch.log(mel + 1e-6)


class FSDDDataset(Dataset):
    def __init__(self, samples: list[Sample], cache: bool = True):
        self.samples = samples
        self.cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._cache_enabled = cache

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        if self._cache_enabled and idx in self.cache:
            return self.cache[idx]
        s = self.samples[idx]
        wav = load_wav(s.path)
        feats = compute_logmel(wav).T  # (frames, n_mels)
        target = torch.tensor([label_to_id(str(s.digit))], dtype=torch.long)
        item = (feats, target)
        if self._cache_enabled:
            self.cache[idx] = item
        return item


def collate(batch):
    feats, targets = zip(*batch)
    feat_lens = torch.tensor([f.shape[0] for f in feats], dtype=torch.long)
    target_lens = torch.tensor([t.shape[0] for t in targets], dtype=torch.long)
    max_t = int(feat_lens.max())
    n_mels = feats[0].shape[1]
    feats_pad = torch.zeros(len(feats), max_t, n_mels)
    for i, f in enumerate(feats):
        feats_pad[i, : f.shape[0]] = f
    targets_cat = torch.cat(targets)
    return feats_pad, feat_lens, targets_cat, target_lens
