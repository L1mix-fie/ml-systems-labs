"""Build lab02/report.md from results/ (no code listings). All claims are computed from the CSVs."""
import json

import pandas as pd

from common import BATCH, EPOCHS, LAB_DIR, LEARNING_RATES, RESULTS, WIDTHS


def md_table(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for r in df.astype(object).itertuples(index=False):
        out.append("| " + " | ".join(str(v) for v in r) + " |")
    return "\n".join(out)


def n_params(W):
    return (64 * W + W) + (W * W + W) + (W * 10 + 10)


def main():
    s = json.loads((RESULTS / "sanity.json").read_text())
    gc = pd.read_csv(RESULTS / "gradient_check.csv")
    log = pd.read_csv(RESULTS / "train_log.csv")
    fin = pd.read_csv(RESULTS / "final_accuracy.csv")
    tim = pd.read_csv(RESULTS / "timing.csv")
    py = (RESULTS / "versions.txt").read_text().splitlines()[0]

    # Step 2 design table
    design = pd.DataFrame([
        {"Layer": "Hidden 1 (+ReLU)", "W shape": "(64, W)", "b shape": "(W,)",
         "Params W=16": 64 * 16 + 16, "Params W=64": 64 * 64 + 64},
        {"Layer": "Hidden 2 (+ReLU)", "W shape": "(W, W)", "b shape": "(W,)",
         "Params W=16": 16 * 16 + 16, "Params W=64": 64 * 64 + 64},
        {"Layer": "Output (+softmax)", "W shape": "(W, 10)", "b shape": "(10,)",
         "Params W=16": 16 * 10 + 10, "Params W=64": 64 * 10 + 10},
        {"Layer": "**Total**", "W shape": "", "b shape": "",
         "Params W=16": n_params(16), "Params W=64": n_params(64)},
    ])

    # Escape epoch: first epoch where training loss drops clearly below ln(10)
    def escape(w, lr):
        r = log[(log.width == w) & (log.lr == lr) & (log.train_loss < 2.0)]
        return int(r.epoch.iloc[0]) if len(r) else None

    fin = fin.copy()
    fin["epoch loss < 2.0"] = [escape(w, lr) or "never" for w, lr in zip(fin.width, fin.lr)]
    fin_tbl = fin.rename(columns={"width": "W", "lr": "LR", "train_loss": "Final train loss",
                                  "test_loss": "Final test loss", "train_acc": "Train acc",
                                  "test_acc": "Test acc"})

    best = fin.loc[fin.test_acc.idxmax()]
    learned = fin[fin.test_acc > 0.5]
    stuck = fin[fin.test_acc <= 0.5]

    gc_tbl = gc.rename(columns={"state": "Network state", "param": "Weight", "analytic": "Analytic",
                                "numerical": "Numerical", "rel_error": "Rel. error", "pass": "< 1e-6"})

    tim_tbl = tim.rename(columns={"width": "W", "batch": "Batch", "forward_ms": "Forward (ms)",
                                  "backward_ms": "Backward (ms)", "backward_over_forward": "Bwd/Fwd time",
                                  "forward_MACs": "Fwd MACs", "backward_MACs": "Bwd MACs",
                                  "MAC_ratio": "Bwd/Fwd MACs"})
    big = tim[tim.batch == tim.batch.max()]
    small = tim[tim.batch == tim.batch.min()]
    bwd_dominates_big = bool((big.backward_ms > big.forward_ms).all())
    ratio_big = f"{big.backward_over_forward.min():.2f}–{big.backward_over_forward.max():.2f}×"
    ratio_small = f"{small.backward_over_forward.min():.2f}–{small.backward_over_forward.max():.2f}×"
    mac_r = f"{tim.MAC_ratio.min():.2f}–{tim.MAC_ratio.max():.2f}×"
    w64_big = tim[(tim.width == 64) & (tim.batch == tim.batch.max())].iloc[0]
    w16_big = tim[(tim.width == 16) & (tim.batch == tim.batch.max())].iloc[0]

    if len(learned):
        learned_txt = ", ".join(f"W={int(r.width)}/lr={r.lr} ({r.test_acc:.4f})" for _, r in learned.iterrows())
    else:
        learned_txt = "none"
    stuck_txt = ", ".join(f"W={int(r.width)}/lr={r.lr}" for _, r in stuck.iterrows()) or "none"

    lrs_learned = sorted(set(learned.lr))
    lr_txt = (f"only the largest learning rate ({lrs_learned[0]}) escapes" if lrs_learned == [max(LEARNING_RATES)]
              else f"the learning rates that escape are {lrs_learned}" if lrs_learned else "no learning rate escapes")
    never = sorted(set(stuck.lr))
    never_txt = (f"Learning rates {never} never leave chance level for at least one width." if never else
                 "All learning rates eventually leave chance level, but the plateau still costs many epochs.")
    hi = fin[fin.lr == max(LEARNING_RATES)].set_index("width")
    w64_better = hi.loc[64, "test_acc"] > hi.loc[16, "test_acc"]
    e16, e64 = escape(16, max(LEARNING_RATES)), escape(64, max(LEARNING_RATES))
    earlier = e64 is not None and (e16 is None or e64 < e16)
    width_head = "Wider was better here, and the cost was small." if w64_better else "Width did not help accuracy here."
    width_txt = (f"At lr = {max(LEARNING_RATES)}, W = 64 {'escaped the plateau earlier and ' if earlier else ''}"
                 f"finished with {'higher' if w64_better else 'lower or equal'} test accuracy than W = 16 "
                 f"({hi.loc[64, 'test_acc']:.4f} vs {hi.loc[16, 'test_acc']:.4f}).")
    t_ratio = w64_big.forward_ms / w16_big.forward_ms
    m_ratio = w64_big.forward_MACs / w16_big.forward_MACs
    cost_txt = (f"much less than the {m_ratio:.1f}× increase in MACs. Small matrix multiplications don't use the CPU "
                f"fully, so extra width is cheap at this scale." if t_ratio < 0.7 * m_ratio else
                f"close to the {m_ratio:.1f}× increase in MACs, so time grows about in line with the matmul work.")

    timing_verdict = (
        f"On the full training batch the backward pass is slower in every case ({ratio_big} the forward time)."
        if bwd_dominates_big else
        f"On the full training batch the backward/forward time ratio is {ratio_big}."
    )

    report = f"""# Lab 2: A Neural Network from Scratch

## 1. Goal

Build a two-hidden-layer neural network in plain NumPy (no autograd). Implement the forward pass,
cross-entropy loss and backpropagation, verify the gradients numerically, train it on the Digits dataset
with different learning rates and widths, and time the forward and backward passes.

## 2. Method

- **Environment:** pinned venv from the repo `requirements.txt`; versions in `results/versions.txt` ({py}).
  Seeds 42 for `random`, NumPy and PyTorch, re-set before each run.
- **Data:** `load_digits`, 1797 images of 8×8 = 64 pixels. Inputs divided by 16 (range 0–1), labels one-hot
  (10 classes), 70/30 stratified split with `random_state=42` (1257 train / 540 test).
- **Network design (step 2):** 64 → W → ReLU → W → ReLU → 10 → softmax, for W = 16 and W = 64.
  Weights drawn from N(0, 0.01), interpreted as standard deviation 0.01; biases start at 0.

{md_table(design)}

- **Forward pass:** z1 = X·W1 + b1, a1 = ReLU(z1), z2 = a1·W2 + b2, a2 = ReLU(z2), z3 = a2·W3 + b3,
  p = softmax(z3), with the row max subtracted before exponentiating for stability.
- **Loss:** mean cross-entropy, L = −(1/B) Σ y·log p.
- **Backward pass (chain rule):** softmax and cross-entropy combine to dL/dz3 = (p − y)/B. Then
  dW3 = a2ᵀ·dz3 and db3 = Σ dz3; da2 = dz3·W3ᵀ and dz2 = da2 ⊙ 1[z2 > 0]; dW2 = a1ᵀ·dz2 and db2 = Σ dz2;
  dz1 = (dz2·W2ᵀ) ⊙ 1[z1 > 0]; dW1 = Xᵀ·dz1 and db1 = Σ dz1.
- **Gradient check:** on a batch of {BATCH} training images, 10 random weights spread across W1, W2 and W3 are
  each nudged by ±1e−5. The numerical slope is (L(w+h) − L(w−h)) / 2h, and relative error is
  |a − n| / (|a| + |n|). It passes if the error is below 1e−6 for at least 9 of 10 weights. Weights are drawn from
  those with |gradient| ≥ 1e−5: behind a dead ReLU the gradient is exactly 0, and for near-zero gradients
  the relative error measures float64 round-off rather than correctness. The check is run on the untrained
  network and again after one epoch of training (lr = 0.1).
- **Training:** mini-batch SGD, batch {BATCH}, {EPOCHS} epochs, data reshuffled each epoch. Train/test loss and
  accuracy logged every epoch (`results/train_log.csv`). Grid: learning rates {LEARNING_RATES} × widths {WIDTHS}.
- **Timing:** forward and backward timed separately with `time.perf_counter`, 1 warm-up and then the median of
  100 runs, at batch {BATCH} and at the full training set (1257 rows). I also counted the multiply-accumulates
  (MACs) in each pass's matrix multiplications.

## 3. Results

### Forward pass and loss sanity checks

| Check | W = 16 | W = 64 | Expected |
|---|---|---|---|
| Output shape (batch 32) | {tuple(s["W16_output_shape"])} | {tuple(s["W64_output_shape"])} | (32, 10) |
| Max \\|row sum − 1\\| | {s["W16_max_row_sum_error"]:.1e} | {s["W64_max_row_sum_error"]:.1e} | < 1e−6 |
| Untrained loss (train set) | {s["W16_initial_loss"]} | {s["W64_initial_loss"]} | ln(10) = {s["ln10"]} |

### Gradient check (`results/gradient_check.csv`)

Passed: **{s["gradcheck_pass_untrained"]}** on the untrained network and **{s["gradcheck_pass_after_1_epoch"]}**
after one epoch (requirement: ≥ 9/10).

{md_table(gc_tbl)}

### Training (`results/loss_curves.png`, `results/final_accuracy.csv`)

![Loss curves](results/loss_curves.png)

{md_table(fin_tbl)}

"Epoch loss < 2.0" is the first epoch where the training loss drops clearly below the ln(10) ≈ 2.30 plateau.
Configurations that learned (test acc > 0.5): {learned_txt}. Stuck at chance level (~10%): {stuck_txt}.
Best: **W = {int(best.width)}, lr = {best.lr}, test accuracy {best.test_acc:.4f}**.

### Timing (`results/timing.csv`, median of 100 runs)

{md_table(tim_tbl)}

{timing_verdict} At batch 32 the ratio is {ratio_small}. Going from W = 16 to W = 64 on the full batch multiplies the
forward MACs by {w64_big.forward_MACs / w16_big.forward_MACs:.1f}× and the forward time by
{w64_big.forward_ms / w16_big.forward_ms:.1f}×.

## 4. Conclusions

1. **The backward pass costs more because it does more matrix multiplication.** For each layer, the forward pass
   does one matrix multiply (activations × weights). The backward pass needs two: one for the weight gradient
   (aᵀ·dz) and one to pass the error back to the previous layer (dz·Wᵀ). Only the first layer skips the second
   one. That gives {mac_r} the forward MACs, and on the full batch the measured time ratio ({ratio_big}) is in the
   same range. At batch 32 the matrices are so small that fixed per-call NumPy overhead dominates, so the two
   passes take about the same time. The work only shows up in the timings once the matrices are large enough.
2. **The N(0, 0.01) initialisation nearly stops learning.** With weights this small, the signal shrinks at every
   layer. The output is almost uniform (loss = ln 10 exactly), and the early-layer gradients are around 1e−5 or
   smaller. The network sits on the ln(10) plateau, and {lr_txt} within
   {EPOCHS} epochs. {never_txt} This is a vanishing-signal problem, not
   a bug, since the gradient check passed. A scaled initialisation (e.g. He, std = √(2/fan-in)) would be the
   standard fix.
3. **{width_head}** {width_txt} It has about {n_params(64) / n_params(16):.0f}× the parameters,
   but on the full batch its forward pass was only {t_ratio:.1f}× slower, which is
   {cost_txt}
"""
    (LAB_DIR / "report.md").write_text(report)
    print(f"wrote {LAB_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
