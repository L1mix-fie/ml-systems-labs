# Lab 1: Environment and First System Measurements

## 1. Goal

Set up a reproducible, pinned Python environment and measure the *system* cost (training time,
single-sample inference latency, model size, peak memory) of two baseline classifiers on the
Breast Cancer Wisconsin (Diagnostic) dataset, then decide which deployment tiers
(Cloud, Edge, Mobile, TinyML) each model fits.

## 2. Method

- **Environment:** isolated venv from `requirements.txt` (pinned versions). `src/print_versions.py`
  writes interpreter and package versions to `results/versions.txt` (run used: Python 3.11.8 (CPython)).
- **Dependency conflict:** the pinned list is unsatisfiable as written (mlflow 2.14.1 requires pyarrow<16,
  but pyarrow==16.1.0 is pinned). I installed with `uv pip install -r requirements.txt --override overrides.txt`,
  forcing pyarrow==16.1.0, so every version matches the manual. mlflow is not used in this lab.
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

| Model | Test acc | Train time (ms, median of 5) | Inference (ms, median of 100) | Size (bytes) | Size (KB) | Peak RSS train (MB) | Peak RSS infer (MB) | RSS increase train / infer (MB) |
|---|---|---|---|---|---|---|---|---|
| LogisticRegression | 0.9415 | 89.723 | 0.043 | 1103 | 1.08 | 324.75 | 324.75 | 0.08 / 0.03 |
| RandomForest | 0.9357 | 50.275 | 0.9402 | 290905 | 284.09 | 324.78 | 324.92 | 0.06 / 0.17 |

**Budgets used:** Cloud mem ≥ 1 GB, ≤ 100 ms, ≤ 500 MB · Edge 256–1024 MB, ≤ 50 ms, ≤ 50 MB ·
Mobile 64–256 MB, ≤ 20 ms, ≤ 10 MB · TinyML ≤ 256 KB, ≤ 10 ms, ≤ 100 KB.

**Deployment fit** (`results/deployment_fit.csv`)

| Model | Cloud | Edge | Mobile | TinyML |
|---|---|---|---|---|
| LogisticRegression | ✅ fits | ✅ fits | ❌ | ❌ |
| RandomForest | ✅ fits | ✅ fits | ❌ | ❌ |

Failed constraints for TinyML — LogisticRegression: memory;
RandomForest: memory and size.

**Where each model can run**

- **Logistic Regression: Cloud and Edge.** Latency (0.043 ms) and size (1.08 KB) are far inside every
  budget, including TinyML. It fails Mobile and TinyML only on memory: the running Python process peaks at
  about 325 MB RSS, above the 256 MB Mobile cap, while the model itself added only 0.03 MB during inference.
  The model is just 30 weights and a bias, so exported to C (a dot product and a sigmoid) it would fit a
  microcontroller easily. As measured in Python, it does not.
- **Random Forest: Cloud and Edge.** Latency (0.94 ms) and size (284.09 KB) are inside the Cloud, Edge
  and Mobile limits, so it also fails Mobile only on process memory (325 MB > 256 MB). It fails TinyML on
  memory and size: the file alone is almost 3× the 100 KB limit, so even a C port of the 100 trees would
  not fit without pruning or fewer trees.
- **What it would take to reach Mobile:** about 70 MB less RSS. Most of the 325 MB is the loaded
  libraries (Python, NumPy, scikit-learn, and PyTorch, which is imported only to set its seed). A runtime
  without PyTorch, or an exported model (e.g. ONNX Runtime), would likely bring both models under 256 MB.

## 4. Conclusions

1. **Accuracy does not separate the models, system cost does.** Both reach similar test accuracy
   (LR 0.9415, RF 0.9357), but the random forest is about 264× larger on disk and about
   22× slower for one prediction. For this dataset the simpler linear model is the better
   deployment choice.
2. **The runtime dominates memory, not the model.** Peak RSS was about 325 MB
   for both models, and the measured increase from the model itself was under a few MB. That memory is the
   Python interpreter and the loaded libraries. It is why both models fail Mobile and TinyML even though their
   latency and (for LR) size would fit, so getting to smaller devices is a runtime/export problem as much as
   a model-size problem.
3. **Measurement method matters.** Single-sample latency for the forest is mostly fixed per-call overhead
   (walking 100 trees and scikit-learn's input checks), not arithmetic, so batching or a compiled runtime
   would change the picture. Using a warm-up and the median of many runs gave stable numbers. A single
   cold run would have included one-time costs and over-stated latency.

_Note: lbfgs may emit a ConvergenceWarning because the spec fixes `max_iter=1000` on unscaled features;
the warning is suppressed and the spec was kept unchanged._
