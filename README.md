# ML Systems Labs

Individual lab repository. Textbook: Reddi, *Introduction to Machine Learning Systems*.

## Setup (Python 3.11.8)

```bash
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Layout

- `labNN/src/` — code
- `labNN/results/` — generated outputs (CSV, versions, models)
- `labNN/report.md` — goal, method, results, conclusions
- `data/` — downloaded datasets (git-ignored; scripts download if missing)

## Lab 1

Run from the repo root:

```bash
python lab01/src/run_all.py
```

This prints versions, trains both baselines, measures system cost, checks deployment budgets, and regenerates `lab01/report.md`.
