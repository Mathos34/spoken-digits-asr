"""Visualizations for the trained CTC model."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import (FSDDDataset, compute_logmel, discover, greedy_ctc_decode,
                      ids_to_str, load_wav, split_by_index)  # noqa: E402
from src.model import TinyCTC  # noqa: E402


def main():
    runs = ROOT / "runs"
    out = ROOT / "assets"
    out.mkdir(exist_ok=True)
    with open(runs / "metrics.json", encoding="utf-8") as f:
        metrics = json.load(f)
    confusion = np.array(metrics["confusion"])

    samples = discover(ROOT / "data")
    _, test_s = split_by_index(samples)
    sample = next(s for s in test_s if s.digit == 7)
    wav = load_wav(sample.path)
    spec = compute_logmel(wav).numpy()

    model = TinyCTC()
    model.load_state_dict(torch.load(runs / "model.pt", map_location="cpu"))
    model.eval()
    feats = torch.from_numpy(spec.T).unsqueeze(0)
    with torch.no_grad():
        logits, new_lens = model(feats, torch.tensor([feats.shape[1]]))
        T = int(new_lens[0])
        pred = ids_to_str(greedy_ctc_decode(logits[0, :T]))

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 3)

    ax_wav = fig.add_subplot(gs[0, 0])
    t = np.arange(wav.shape[0]) / 8000.0
    ax_wav.plot(t, wav.numpy(), color="#1f77b4", linewidth=0.5)
    ax_wav.set_title(f"Waveform (digit {sample.digit}, speaker {sample.speaker})")
    ax_wav.set_xlabel("seconds"); ax_wav.set_ylabel("amplitude")
    ax_wav.grid(alpha=0.3)

    ax_spec = fig.add_subplot(gs[0, 1])
    im = ax_spec.imshow(spec, origin="lower", aspect="auto", cmap="magma")
    ax_spec.set_title(f"Log-mel spectrogram\nGreedy CTC prediction: '{pred}'")
    ax_spec.set_xlabel("frame (10 ms hop)"); ax_spec.set_ylabel("mel bin")
    fig.colorbar(im, ax=ax_spec, fraction=0.046)

    ax_curve = fig.add_subplot(gs[0, 2])
    ax_curve.plot(metrics["history"]["epoch"], metrics["history"]["train_loss"],
                  marker="o", color="#1f77b4", label="train loss")
    ax_curve.set_xlabel("epoch"); ax_curve.set_ylabel("CTC loss", color="#1f77b4")
    ax_curve.tick_params(axis="y", labelcolor="#1f77b4")
    ax_cer = ax_curve.twinx()
    ax_cer.plot(metrics["history"]["epoch"], [c * 100 for c in metrics["history"]["test_cer"]],
                marker="s", color="#d62728", label="test CER")
    ax_cer.set_ylabel("test CER (%)", color="#d62728")
    ax_cer.tick_params(axis="y", labelcolor="#d62728")
    ax_curve.set_title("Training curves")
    ax_curve.grid(alpha=0.3)

    ax_conf = fig.add_subplot(gs[1, :])
    norm = confusion / confusion.sum(axis=1, keepdims=True).clip(min=1)
    im = ax_conf.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax_conf.set_xticks(range(10)); ax_conf.set_yticks(range(10))
    ax_conf.set_xlabel("predicted digit"); ax_conf.set_ylabel("true digit")
    ax_conf.set_title(f"Confusion matrix on test set (CER = {metrics['final_test_cer']*100:.2f}%)")
    for i in range(10):
        for j in range(10):
            v = confusion[i, j]
            if v > 0:
                ax_conf.text(j, i, str(v), ha="center", va="center",
                             color="white" if norm[i, j] > 0.5 else "black", fontsize=9)
    fig.colorbar(im, ax=ax_conf, fraction=0.025)

    fig.suptitle("spoken-digits-asr: Conv1D + BiLSTM + CTC on FSDD", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "result.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out / 'result.png'}")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    import matplotlib.patches as patches

    def box(x, y, w, h, label, color):
        ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                            edgecolor=color, facecolor="white", linewidth=2))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=10, color=color)

    box(0.02, 0.40, 0.13, 0.30, "WAV\n(8 kHz)", "#444")
    box(0.17, 0.40, 0.14, 0.30, "Log-mel\n(64 bands)", "#888")
    box(0.33, 0.40, 0.14, 0.30, "Conv1D x2\n(stride 2)", "#1f77b4")
    box(0.49, 0.40, 0.14, 0.30, "BiLSTM x2\n(hidden 32)", "#ff7f0e")
    box(0.65, 0.40, 0.14, 0.30, "Linear\n(11 classes)", "#2ca02c")
    box(0.81, 0.40, 0.16, 0.30, "CTC loss\n+ greedy decode", "#d62728")

    for x in [0.15, 0.31, 0.47, 0.63, 0.79]:
        ax.annotate("", xy=(x + 0.02, 0.55), xytext=(x, 0.55), arrowprops=dict(arrowstyle="->", color="#555"))

    ax.text(0.5, 0.92, "Architecture: tiny end-to-end ASR (~50k params)",
            ha="center", fontsize=12, weight="bold")
    ax.text(0.5, 0.10, "Trained from scratch with CTC loss; vocabulary = digits 0..9 + blank.",
            ha="center", fontsize=9, color="#555")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out / "architecture.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out / 'architecture.png'}")


if __name__ == "__main__":
    main()
