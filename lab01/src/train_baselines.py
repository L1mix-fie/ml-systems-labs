"""Train both baselines on a 70/30 stratified split; save test accuracy (4 dp)."""
import csv

from sklearn.metrics import accuracy_score

from common import MODELS, RESULTS, load_split, set_seeds


def main():
    set_seeds()
    X_tr, X_te, y_tr, y_te = load_split()
    rows = []
    for name, make in MODELS.items():
        model = make().fit(X_tr, y_tr)
        acc = accuracy_score(y_te, model.predict(X_te))
        rows.append({"model": name, "test_accuracy": f"{acc:.4f}"})
        print(f"{name}: {acc:.4f}")
    with open(RESULTS / "baseline_accuracy.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "test_accuracy"])
        w.writeheader()
        w.writerows(rows)
    print(f"train={len(y_tr)} test={len(y_te)}")


if __name__ == "__main__":
    main()
