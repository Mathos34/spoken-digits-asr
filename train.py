"""Train a tiny Conv1D + BiLSTM CTC model on Free Spoken Digit Dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import (
    BLANK_ID,
    FSDDDataset,
    collate,
    discover,
    greedy_ctc_decode,
    ids_to_str,
    split_by_index,
)
from src.metrics import cer
from src.model import TinyCTC, count_params

SEED = 42


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(model: TinyCTC, loader: DataLoader) -> tuple[float, list[str], list[str]]:
    model.eval()
    refs, hyps = [], []
    with torch.no_grad():
        for feats, feat_lens, targets, target_lens in loader:
            logits, new_lens = model(feats, feat_lens)
            t_offset = 0
            for i in range(feats.shape[0]):
                tlen = int(target_lens[i])
                ref_ids = targets[t_offset: t_offset + tlen].tolist()
                t_offset += tlen
                refs.append(ids_to_str(ref_ids))
                T = int(new_lens[i])
                hyp_ids = greedy_ctc_decode(logits[i, :T])
                hyps.append(ids_to_str(hyp_ids))
    return cer(refs, hyps), refs, hyps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", type=str, default="runs")
    args = parser.parse_args()

    set_seed(SEED)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = discover(Path(args.data))
    train_s, test_s = split_by_index(samples)
    print(f"Train: {len(train_s)} samples, Test: {len(test_s)} samples")

    train_ds = FSDDDataset(train_s)
    test_ds = FSDDDataset(test_s)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate, num_workers=0)

    model = TinyCTC()
    print(f"Model: {count_params(model)} params")
    ctc = nn.CTCLoss(blank=BLANK_ID, zero_infinity=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    history: dict[str, list[float]] = {"epoch": [], "train_loss": [], "test_cer": []}
    best_cer = float("inf")
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        n = 0
        for feats, feat_lens, targets, target_lens in tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}"):
            logits, new_lens = model(feats, feat_lens)
            log_probs = logits.log_softmax(dim=-1).transpose(0, 1)  # (T, B, V)
            loss = ctc(log_probs, targets, new_lens, target_lens)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            running += loss.item() * feats.shape[0]
            n += feats.shape[0]
        train_loss = running / n
        test_cer, _, _ = evaluate(model, test_loader)
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["test_cer"].append(test_cer)
        print(f"  epoch {epoch+1}: train_loss={train_loss:.4f} test_cer={test_cer*100:.2f}%")
        if test_cer < best_cer:
            best_cer = test_cer
            torch.save(model.state_dict(), out_dir / "model.pt")
    if not (out_dir / "model.pt").exists():
        torch.save(model.state_dict(), out_dir / "model.pt")

    print("Final eval...")
    model.load_state_dict(torch.load(out_dir / "model.pt", map_location="cpu"))
    final_cer, refs, hyps = evaluate(model, test_loader)
    correct = sum(1 for r, h in zip(refs, hyps, strict=False) if r == h)
    accuracy = correct / max(1, len(refs))

    confusion = np.zeros((10, 10), dtype=np.int64)
    null_predictions = 0
    for r, h in zip(refs, hyps, strict=False):
        ri = int(r) if r else None
        if not h:
            null_predictions += 1
            continue
        hi = int(h[0])
        if ri is not None:
            confusion[ri, hi] += 1

    metrics = {
        "n_train": len(train_s),
        "n_test": len(test_s),
        "epochs": args.epochs,
        "params": count_params(model),
        "best_test_cer": best_cer,
        "final_test_cer": final_cer,
        "test_accuracy_per_utterance": accuracy,
        "null_predictions": null_predictions,
        "history": history,
        "confusion": confusion.tolist(),
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Best CER: {best_cer*100:.2f}%, exact-utterance accuracy: {accuracy*100:.1f}%")


if __name__ == "__main__":
    main()
