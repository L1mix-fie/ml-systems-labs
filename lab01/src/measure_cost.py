"""Measure system cost for both baselines.

- Training time: 1 warm-up fit, then 5 timed fits -> median.
- Single-sample inference: 1 warm-up, then 100 timed predicts -> median.
- Model size: joblib.dump to results/model.joblib -> bytes and KB
  (each model is also kept as results/model_<name>.joblib).
- Peak memory: psutil RSS sampled every 1 ms by a background thread during
  training and during inference. Reports peak process RSS and the increase
  over the RSS measured just before the phase started.
"""
import csv
import shutil
import statistics
import threading
import time

import joblib
import psutil

from common import MODELS, RESULTS, load_split, set_seeds

N_TRAIN_RUNS = 5
N_INFER_RUNS = 100
MB = 1024 * 1024


class PeakRSS:
    """Context manager: track peak RSS of this process while the block runs."""

    def __init__(self, interval=0.001):
        self.proc = psutil.Process()
        self.interval = interval

    def _poll(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self.proc.memory_info().rss)
            time.sleep(self.interval)

    def __enter__(self):
        self.baseline = self.proc.memory_info().rss
        self.peak = self.baseline
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._poll, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()
        self.peak = max(self.peak, self.proc.memory_info().rss)

    @property
    def peak_mb(self):
        return self.peak / MB

    @property
    def delta_mb(self):
        return (self.peak - self.baseline) / MB


def measure(name, make, X_tr, y_tr, X_te):
    # Training
    with PeakRSS() as mem_train:
        make().fit(X_tr, y_tr)  # warm-up
        times = []
        for _ in range(N_TRAIN_RUNS):
            set_seeds()
            m = make()
            t0 = time.perf_counter()
            m.fit(X_tr, y_tr)
            times.append(time.perf_counter() - t0)
    model = m
    train_ms = statistics.median(times) * 1000

    # Single-sample inference
    x = X_te[:1]
    with PeakRSS() as mem_inf:
        model.predict(x)  # warm-up
        lat = []
        for _ in range(N_INFER_RUNS):
            t0 = time.perf_counter()
            model.predict(x)
            lat.append(time.perf_counter() - t0)
    infer_ms = statistics.median(lat) * 1000

    # Model size
    path = RESULTS / "model.joblib"
    joblib.dump(model, path)
    size_b = path.stat().st_size
    shutil.copy(path, RESULTS / f"model_{name}.joblib")

    return {
        "model": name,
        "train_time_ms_median": round(train_ms, 3),
        "infer_latency_ms_median": round(infer_ms, 4),
        "model_size_bytes": size_b,
        "model_size_kb": round(size_b / 1024, 2),
        "train_peak_rss_mb": round(mem_train.peak_mb, 2),
        "train_rss_increase_mb": round(mem_train.delta_mb, 2),
        "infer_peak_rss_mb": round(mem_inf.peak_mb, 2),
        "infer_rss_increase_mb": round(mem_inf.delta_mb, 2),
    }


def main():
    set_seeds()
    X_tr, X_te, y_tr, _ = load_split()
    rows = []
    for name, make in MODELS.items():
        r = measure(name, make, X_tr, y_tr, X_te)
        print(r)
        rows.append(r)
    with open(RESULTS / "system_cost.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
