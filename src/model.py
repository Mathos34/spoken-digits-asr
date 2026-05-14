"""Tiny Conv1D + BiLSTM acoustic model with CTC head."""
from __future__ import annotations

import torch
import torch.nn as nn

from .data import N_MELS, VOCAB_SIZE


class TinyCTC(nn.Module):
    def __init__(self, n_mels: int = N_MELS, hidden: int = 64, vocab_size: int = VOCAB_SIZE):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_mels, hidden, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(hidden, hidden // 2, num_layers=2, batch_first=True, bidirectional=True)
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, n_mels) -> conv expects (B, n_mels, T)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)  # (B, T', hidden)
        new_lengths = ((lengths + 1) // 2 + 1) // 2
        new_lengths = torch.clamp(new_lengths, max=x.shape[1])
        x, _ = self.lstm(x)
        logits = self.head(x)
        return logits, new_lengths


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
