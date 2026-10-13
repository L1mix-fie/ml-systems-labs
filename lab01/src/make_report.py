"""Build lab01/report.md from the CSVs in results/ (no code listings in the report)."""
import pandas as pd

from common import LAB_DIR, RESULTS


def md_table(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def yn(b):
    return "yes" if b else "no"


def main():
    acc = pd.read_csv(RESULTS / "baseline_accuracy.csv", dtype={"test_accuracy": str})
    cost = pd.read_csv(RESULTS / "system_cost.csv")
    fit = pd.read_csv(RESULTS / "deployment_fit.csv")
    py = (RESULTS / "versions.txt").read_text().splitlines()[0]

    lr = cost[cost.model == "LogisticRegression"].iloc[0]
    rf = cost[cost.model == "RandomForest"].iloc[0]
    a_lr = acc[acc.model == "LogisticRegression"].test_accuracy.iloc[0]
    a_rf = acc[acc.model == "RandomForest"].test_accuracy.iloc[0]

    measure = pd.DataFrame({
        "Model": cost.model,
        "Test acc": [acc.set_index("model").test_accuracy[m] for m in cost.model],
        "Train time (ms, median of 5)": cost.train_time_ms_median,
        "Inference (ms, median of 100)": cost.infer_latency_ms_median,
        "Size (bytes)": cost.model_size_bytes,
        "Size (KB)": cost.model_size_kb,
        "Peak RSS train (MB)": cost.train_peak_rss_mb,
        "Peak RSS infer (MB)": cost.infer_peak_rss_mb,
        "RSS increase train / infer (MB)": [
            f"{a} / {b}" for a, b in zip(cost.train_rss_increase_mb, cost.infer_rss_increase_mb)
        ],
    })

    grid = fit.pivot(index="model", columns="tier", values="fits")[["Cloud", "Edge", "Mobile", "TinyML"]]
    grid = grid.map(lambda b: "✅ fits" if b else "❌") if hasattr(grid, "map") else grid.applymap(lambda b: "✅ fits" if b else "❌")
    grid = grid.reset_index().rename(columns={"model": "Model"})

    def fails(model, tier):
        r = fit[(fit.model == model) & (fit.tier == tier)].iloc[0]
        f = [n for n, ok in (("memory", r.memory_ok), ("latency", r.latency_ok), ("size", r.size_ok)) if not ok]
        return " and ".join(f) if f else "none"

    speedup = rf.infer_latency_ms_median / lr.infer_latency_ms_median
    size_ratio = rf.model_size_bytes / lr.model_size_bytes

    report = f"""# Lab 1: Environment and First System Measurements

## 1. Goal

Set up a reproducible, pinned Python environment and measure the *system* cost (training time,
single-sample inference latency, model size, peak memory) of two baseline classifiers on the
Breast Cancer Wisconsin (Diagnostic) dataset, then decide which deployment tiers
(Cloud, Edge, Mobile, TinyML) each model fits.

## 2. Method

- **Environment:** isolated venv from `requirements.txt` (pinned versions). `src/print_versions.py`
  writes interpreter and package versions to `results/versions.txt` (run used: {py}).
- **Data:** `load_breast_cancer(return_X_y=True)`, 569 samples × 30 features, 70/30 stratified split
  with `random_state=42` (398 train / 171 test). Seeds set for `random`, NumPy and PyTorch (42).
- **Models:** `LogisticRegression(max_iter=1000, random_state=42)` and
  `RandomForestClassifier(n_estimators=100, random_state=42)`, default hyperparameters otherwise,
  no feature scaling (as specified).
- **Training time:** 1 warm-up fit, then 5 timed fits with `time.perf_counter`; median reported.
- **Inference latency:** one test sample, 1 warm-up call, then 100 timed `predict` calls; median reported.
- **Model size:** `joblib.dump` to `results/model.joblib`, file size in bytes and KB
  (each model also kept as `results/model_<name>.joblib`).
- **Peak memory:** a background thread samples `psutil.Process().memory_info().rss` every 1 ms during the
  training phase and the inference phase. I report the peak process RSS and the increase over the RSS
  just before the phase started.
- **Budget check:** a model fits a tier only if peak RSS ≤ tier memory upper bound (Cloud has no upper bound),
  median latency ≤ tier latency, and model file ≤ tier size limit. Full table in `results/deployment_fit.csv`.

## 3. Results

**Measurements** (`results/baseline_accuracy.csv`, `results/system_cost.csv`)

{md_table(measure)}

**Budgets used:** Cloud mem ≥ 1 GB, ≤ 100 ms, ≤ 500 MB · Edge 256–1024 MB, ≤ 50 ms, ≤ 50 MB ·
Mobile 64–256 MB, ≤ 20 ms, ≤ 10 MB · TinyML ≤ 256 KB, ≤ 10 ms, ≤ 100 KB.

**Deployment fit** (`results/deployment_fit.csv`)

{md_table(grid)}

Failed constraints for TinyML — LogisticRegression: {fails("LogisticRegression", "TinyML")};
RandomForest: {fails("RandomForest", "TinyML")}.

**Where each model can run**

- **Logistic Regression:** Cloud {yn(fit[(fit.model=="LogisticRegression")&(fit.tier=="Cloud")].fits.iloc[0])},
  Edge {yn(fit[(fit.model=="LogisticRegression")&(fit.tier=="Edge")].fits.iloc[0])},
  Mobile {yn(fit[(fit.model=="LogisticRegression")&(fit.tier=="Mobile")].fits.iloc[0])}.
  Latency ({lr.infer_latency_ms_median} ms) and size ({lr.model_size_kb} KB) are far inside every budget,
  including TinyML. It fails TinyML only on memory: the Python + scikit-learn process needs about
  {lr.infer_peak_rss_mb:.0f} MB of RSS, while the model itself added only {lr.infer_rss_increase_mb} MB during inference.
  The model is just 30 weights and a bias, so exported to C (a dot product and a sigmoid) it would fit a
  microcontroller easily. As measured in Python, it does not.
- **Random Forest:** Cloud, Edge and Mobile pass. Latency ({rf.infer_latency_ms_median} ms) is under the
  Mobile 20 ms limit, and the {rf.model_size_kb} KB file is under 10 MB. It fails TinyML on
  {fails("RandomForest", "TinyML")}: the file alone is above the 100 KB limit, so even a C port of the
  100 trees would not fit without pruning or fewer trees.

## 4. Conclusions

1. **Accuracy does not separate the models, system cost does.** Both reach similar test accuracy
   (LR {a_lr}, RF {a_rf}), but the random forest is about {size_ratio:.0f}× larger on disk and about
   {speedup:.0f}× slower for one prediction. For this dataset the simpler linear model is the better
   deployment choice.
2. **The runtime dominates memory, not the model.** Peak RSS was about {max(lr.infer_peak_rss_mb, rf.infer_peak_rss_mb):.0f} MB
   for both models, and the measured increase from the model itself was under a few MB. That memory is the
   Python interpreter, NumPy and scikit-learn. This is why neither model fits TinyML as-is, and why getting to
   TinyML is a runtime/export problem as well as a model-size problem.
3. **Measurement method matters.** Single-sample latency for the forest is mostly fixed per-call overhead
   (walking 100 trees and scikit-learn's input checks), not arithmetic, so batching or a compiled runtime
   would change the picture. Using a warm-up and the median of many runs gave stable numbers. A single
   cold run would have included one-time costs and over-stated latency.

_Note: lbfgs may emit a ConvergenceWarning because the spec fixes `max_iter=1000` on unscaled features;
the warning is suppressed and the spec was kept unchanged._
"""
    (LAB_DIR / "report.md").write_text(report)
    print(f"wrote {LAB_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
