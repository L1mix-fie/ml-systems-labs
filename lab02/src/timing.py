"""Step 7: time forward and backward passes separately (1 warm-up, median of 100 runs).

Measured for both widths at batch 32 (training batch) and on the full training set (1257 rows).
Timing does not depend on weight values (dense NumPy matmuls), so freshly initialised weights are used.
Also reports the multiply-accumulate (MAC) count of the matrix multiplications in each pass.
"""
import csv
import statistics
import time

from common import BATCH, RESULTS, load_data, set_seeds
from nn import backward, forward, init_params

RUNS = 100


def median_ms(fn):
    fn()  # warm-up
    ts = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1000


def macs(B, W, n_in=64, n_out=10):
    fwd = B * (n_in * W + W * W + W * n_out)
    # backward: weight grads for all 3 layers + input grads for layers 3 and 2 (not layer 1)
    bwd = B * (n_in * W + W * W + W * n_out) + B * (W * n_out + W * W)
    return fwd, bwd


def main():
    set_seeds()
    X_tr, _, Y_tr, _, *_ = load_data()
    rows = []
    for width in (16, 64):
        set_seeds()
        params = init_params(width)
        for B in (BATCH, len(X_tr)):
            X, Y = X_tr[:B], Y_tr[:B]
            p, cache = forward(params, X)
            f_ms = median_ms(lambda: forward(params, X))
            b_ms = median_ms(lambda: backward(params, p, Y, cache))
            f_mac, b_mac = macs(B, width)
            rows.append({"width": width, "batch": B,
                         "forward_ms": round(f_ms, 4), "backward_ms": round(b_ms, 4),
                         "backward_over_forward": round(b_ms / f_ms, 2),
                         "forward_MACs": f_mac, "backward_MACs": b_mac,
                         "MAC_ratio": round(b_mac / f_mac, 2)})
            print(rows[-1])
    with open(RESULTS / "timing.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
