# ML Systems Labs

Individual lab repository. Textbook: Reddi, *Introduction to Machine Learning Systems*.

## Setup (Python 3.11.8)

```bash
uv venv --python 3.11.8 .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt --override overrides.txt
```

The pinned list in the lab manual conflicts (mlflow 2.14.1 requires pyarrow<16, but pyarrow==16.1.0 is pinned).
`overrides.txt` forces pyarrow==16.1.0 so every installed version matches the manual.

## Layout

- `labNN/src/` — code
- `labNN/results/` — generated outputs (CSVs, plots, versions, models)
- `labNN/report.md` — goal, method, results, conclusions
- `data/` — downloaded datasets (git-ignored; scripts download if missing)

Each lab is run from the repo root with `python labNN/src/run_all.py`, which runs every step in order,
saves `results/versions.txt`, and regenerates `labNN/report.md` from the results.
Note: re-running overwrites `report.md`, including any manual edits.

## Lab 1 — Environment and first system measurements

Dataset: Breast Cancer Wisconsin (`load_breast_cancer`).

```bash
python lab01/src/run_all.py
```

Trains Logistic Regression and Random Forest baselines, measures training time, single-sample inference
latency, model size and peak memory, and checks each model against Cloud, Edge, Mobile and TinyML budgets.

| Script | Step |
|---|---|
| `print_versions.py` | Python and package versions → `results/versions.txt` |
| `train_baselines.py` | 70/30 stratified split, both baselines → `results/baseline_accuracy.csv` |
| `measure_cost.py` | Timing, latency, size, peak RSS → `results/system_cost.csv`, `results/model.joblib` |
| `budget_fit.py` | Deployment-tier check → `results/deployment_fit.csv` |

**Key findings** (full conclusions in [`lab01/report.md`](lab01/report.md)):
1. Both models reach similar accuracy (LR 0.9415, RF 0.9357), but the random forest is ~264× larger on disk and
   ~22× slower per prediction, so the linear model is the better deployment choice.
2. Memory is dominated by the Python runtime (~325 MB peak RSS), not the models (< 1 MB added). Both models
   therefore fit only Cloud and Edge; they fail Mobile (256 MB cap) and TinyML on memory, and RF also fails TinyML on size.
3. Single-sample latency is mostly fixed per-call overhead; a warm-up plus the median of many runs was needed for
   stable measurements.

## Lab 2 — A neural network from scratch

Dataset: Optical Recognition of Handwritten Digits (`load_digits`).

```bash
python lab02/src/run_all.py
```

A two-hidden-layer MLP (64 → W → ReLU → W → ReLU → 10 → softmax) written in plain NumPy, with forward pass,
cross-entropy loss and hand-derived backpropagation. It verifies gradients numerically, trains over learning
rates {0.001, 0.01, 0.1} × widths {16, 64}, and times the forward and backward passes.

| Script | Step |
|---|---|
| `nn.py` | Network: init, forward, loss, backward, SGD step |
| `sanity_checks.py` | Output shape, softmax row sums, initial loss ≈ ln(10), numerical gradient check → `results/sanity.json`, `results/gradient_check.csv` |
| `train.py` | Mini-batch SGD grid → `results/train_log.csv`, `results/final_accuracy.csv`, `results/loss_curves.png` |
| `timing.py` | Forward vs backward timing, median of 100 runs → `results/timing.csv` |

**Key findings** (full conclusions in [`lab02/report.md`](lab02/report.md)):
1. The backward pass dominates (1.38–1.98× the forward time on the full batch): it does two matrix multiplications
   per layer (weight gradient and error propagation) versus one in the forward pass, which is 1.29–1.54× the MACs.
2. The required N(0, 0.01) initialisation shrinks the signal so much that the network sits at the ln(10) plateau.
   Only lr = 0.1 escapes within 50 epochs; lr = 0.001 and 0.01 stay at 10% accuracy. Gradients were verified
   correct (10/10 checks), so this is vanishing signal, not a bug.
3. Wider was better: W = 64 reached 94.44% test accuracy versus 67.22% for W = 16, with ~6× the parameters but
   only ~2× the forward time.

Shared in both labs: `common.py` (seeds, data, paths), `make_report.py` (builds the report), `run_all.py`.
