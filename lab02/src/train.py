"""Step 6: mini-batch SGD, batch 32, 50 epochs, shuffle each epoch, per-epoch logging.

Grid: learning rates [0.001, 0.01, 0.1] x widths [16, 64]. Each run re-seeds (42) so runs differ
only in lr and width. Outputs results/train_log.csv, results/final_accuracy.csv, results/loss_curves.png.
"""
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import BATCH, EPOCHS, LEARNING_RATES, RESULTS, WIDTHS, load_data, set_seeds
from nn import backward, cross_entropy, forward, init_params, predict, sgd_step


def train_one(lr, width, X_tr, Y_tr, X_te, Y_te, y_tr, y_te, log):
    set_seeds()
    params = init_params(width)
    n = len(X_tr)
    for epoch in range(1, EPOCHS + 1):
        idx = np.random.permutation(n)
        batch_losses = []
        for s in range(0, n, BATCH):
            b = idx[s:s + BATCH]
            p, cache = forward(params, X_tr[b])
            batch_losses.append(cross_entropy(p, Y_tr[b]) * len(b))
            sgd_step(params, backward(params, p, Y_tr[b], cache), lr)
        train_loss = sum(batch_losses) / n
        test_loss = cross_entropy(forward(params, X_te)[0], Y_te)
        train_acc = float((predict(params, X_tr) == y_tr).mean())
        test_acc = float((predict(params, X_te) == y_te).mean())
        log.append({"lr": lr, "width": width, "epoch": epoch, "train_loss": round(train_loss, 6),
                    "test_loss": round(test_loss, 6), "train_acc": round(train_acc, 4),
                    "test_acc": round(test_acc, 4)})
        if epoch in (1, 10, 25, 50):
            print(f"lr={lr:<6} W={width:<3} epoch {epoch:>2}: train_loss={train_loss:.4f} "
                  f"test_acc={test_acc:.4f}")
    return params


def main():
    X_tr, X_te, Y_tr, Y_te, y_tr, y_te = load_data()
    log = []
    for width in WIDTHS:
        for lr in LEARNING_RATES:
            train_one(lr, width, X_tr, Y_tr, X_te, Y_te, y_tr, y_te, log)

    with open(RESULTS / "train_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(log[0]))
        w.writeheader()
        w.writerows(log)

    final = [r for r in log if r["epoch"] == EPOCHS]
    with open(RESULTS / "final_accuracy.csv", "w", newline="") as f:
        cols = ["width", "lr", "train_loss", "test_loss", "train_acc", "test_acc"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows({c: r[c] for c in cols} for r in final)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    colors = {0.001: "#1f77b4", 0.01: "#ff7f0e", 0.1: "#2ca02c"}
    for ax, width in zip(axes, WIDTHS):
        for lr in LEARNING_RATES:
            rows = [r for r in log if r["width"] == width and r["lr"] == lr]
            ep = [r["epoch"] for r in rows]
            ax.plot(ep, [r["train_loss"] for r in rows], color=colors[lr], label=f"lr={lr} train")
            ax.plot(ep, [r["test_loss"] for r in rows], color=colors[lr], ls="--", label=f"lr={lr} test")
        ax.axhline(np.log(10), color="gray", lw=0.8, ls=":")
        ax.set_title(f"Hidden width W = {width}")
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Cross-entropy loss")
    axes[1].legend(fontsize=8, loc="upper right")
    fig.suptitle("Digits MLP (64-W-W-10), SGD batch 32. Dotted line = ln(10)")
    fig.tight_layout()
    fig.savefig(RESULTS / "loss_curves.png", dpi=150)
    print("saved loss_curves.png")


if __name__ == "__main__":
    main()
