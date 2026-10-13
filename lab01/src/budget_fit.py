"""Check each model against the Cloud / Edge / Mobile / TinyML budgets.

Memory compared = peak process RSS over training and inference (what the
device must actually provide to run this Python model as-is).
A tier passes on memory if peak RSS <= the tier's upper bound (Cloud has none).
"""
import csv

import pandas as pd

from common import RESULTS

KB, MB, GB = 1 / 1024, 1.0, 1024.0  # in MB

# tier: (memory upper bound MB or None, latency ms, model size MB)
BUDGETS = {
    "Cloud": (None, 100, 500),
    "Edge": (1024, 50, 50),
    "Mobile": (256, 20, 10),
    "TinyML": (256 * KB, 10, 100 * KB),
}


def main():
    cost = pd.read_csv(RESULTS / "system_cost.csv")
    rows = []
    for _, r in cost.iterrows():
        mem = max(r.train_peak_rss_mb, r.infer_peak_rss_mb)
        lat = r.infer_latency_ms_median
        size = r.model_size_bytes / (1024 * 1024)
        for tier, (mem_cap, lat_cap, size_cap) in BUDGETS.items():
            mem_ok = mem_cap is None or mem <= mem_cap
            lat_ok = lat <= lat_cap
            size_ok = size <= size_cap
            rows.append({
                "model": r.model,
                "tier": tier,
                "peak_rss_mb": round(mem, 2),
                "memory_ok": mem_ok,
                "latency_ms": lat,
                "latency_ok": lat_ok,
                "size_kb": r.model_size_kb,
                "size_ok": size_ok,
                "fits": mem_ok and lat_ok and size_ok,
            })
    with open(RESULTS / "deployment_fit.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
