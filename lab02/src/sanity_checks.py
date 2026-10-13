"""Steps 3-5: forward-pass checks, initial-loss check, numerical gradient check.

Gradient check: pick 10 random weights spread over W1, W2, W3; nudge each by +/-1e-5;
numerical slope = (L(w+h) - L(w-h)) / 2h; relative error = |a - n| / max(|a| + |n|, 1e-12).
Pass if relative error < 1e-6 for at least 9 of 10.

Run on (a) the untrained network and (b) the network after 1 epoch of training at lr=0.1,
so the backward pass is verified both at initialisation and on non-trivial weights.
"""
import csv
import json
import math

import numpy as np

from common import BATCH, RESULTS, load_data, set_seeds
from nn import backward, cross_entropy, forward, init_params, loss_fn, sgd_step

H = 1e-5
N_CHECK = 10
TOL = 1e-6


MIN_GRAD = 1e-5


def pick_weights(grads, n):
    """n random (matrix, i, j) picks, cycling over W1, W2, W3 so every layer is tested.

    Only weights with |analytic grad| >= MIN_GRAD are eligible: a weight behind a dead ReLU has
    gradient exactly 0, and for near-zero gradients relative error measures float round-off,
    not correctness. If a layer has no eligible weight, its largest-gradient weights are used.
    """
    names = ["W1", "W2", "W3"]
    picks = []
    for k in range(n):
        name = names[k % 3]
        g = np.abs(grads[name])
        cand = np.argwhere(g >= MIN_GRAD)
        if len(cand) == 0:
            cand = np.argwhere(g >= np.sort(g.ravel())[-5])
        for _ in range(100):  # avoid picking the same weight twice
            i, j = cand[np.random.randint(len(cand))]
            if (name, int(i), int(j)) not in picks:
                break
        picks.append((name, int(i), int(j)))
    return picks


def grad_check(params, X, Y, label):
    p, cache = forward(params, X)
    grads = backward(params, p, Y, cache)
    rows = []
    for name, i, j in pick_weights(grads, N_CHECK):
        w = params[name]
        old = w[i, j]
        w[i, j] = old + H
        lp = loss_fn(params, X, Y)
        w[i, j] = old - H
        lm = loss_fn(params, X, Y)
        w[i, j] = old
        num = (lp - lm) / (2 * H)
        ana = grads[name][i, j]
        rel = abs(ana - num) / max(abs(ana) + abs(num), 1e-12)
        rows.append({"state": label, "param": f"{name}[{i},{j}]", "analytic": f"{ana:.6e}",
                     "numerical": f"{num:.6e}", "rel_error": f"{rel:.2e}", "pass": bool(rel < TOL)})
    return rows


def main():
    set_seeds()
    X_tr, X_te, Y_tr, Y_te, *_ = load_data()
    out = {}

    for width in (16, 64):
        set_seeds()
        params = init_params(width)
        p, _ = forward(params, X_tr[:BATCH])
        out[f"W{width}_output_shape"] = list(p.shape)
        out[f"W{width}_max_row_sum_error"] = float(np.abs(p.sum(axis=1) - 1).max())
        out[f"W{width}_initial_loss"] = round(cross_entropy(*forward(params, X_tr)[:1], Y_tr), 4)
    out["ln10"] = round(math.log(10), 4)

    # Gradient check on W=16 (a batch of 32 training samples)
    set_seeds()
    params = init_params(16)
    Xb, Yb = X_tr[:BATCH], Y_tr[:BATCH]
    rows = grad_check(params, Xb, Yb, "untrained")

    # Train one epoch at lr=0.1, then check again
    idx = np.random.permutation(len(X_tr))
    for s in range(0, len(idx), BATCH):
        b = idx[s:s + BATCH]
        p, cache = forward(params, X_tr[b])
        sgd_step(params, backward(params, p, Y_tr[b], cache), 0.1)
    rows += grad_check(params, Xb, Yb, "after 1 epoch")

    for state in ("untrained", "after 1 epoch"):
        n = sum(r["pass"] for r in rows if r["state"] == state)
        out[f"gradcheck_pass_{state.replace(' ', '_')}"] = f"{n}/{N_CHECK}"
        out[f"gradcheck_ok_{state.replace(' ', '_')}"] = bool(n >= 9)

    with open(RESULTS / "gradient_check.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (RESULTS / "sanity.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
