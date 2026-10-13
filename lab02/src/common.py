"""Shared helpers for Lab 2: seeding, data, paths."""
import random
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

SEED = 42
LAB_DIR = Path(__file__).resolve().parents[1]
RESULTS = LAB_DIR / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

LEARNING_RATES = [0.001, 0.01, 0.1]
WIDTHS = [16, 64]
EPOCHS = 50
BATCH = 32


def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def one_hot(y, k=10):
    out = np.zeros((len(y), k))
    out[np.arange(len(y)), y] = 1.0
    return out


def load_data():
    """Digits scaled to [0, 1] (divide by 16), one-hot labels, 70/30 stratified split."""
    X, y = load_digits(return_X_y=True)
    X = X.astype(np.float64) / 16.0
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)
    return X_tr, X_te, one_hot(y_tr), one_hot(y_te), y_tr, y_te
