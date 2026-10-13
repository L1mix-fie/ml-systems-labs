"""Two-hidden-layer MLP in pure NumPy.

64 -> W -> ReLU -> W -> ReLU -> 10 -> softmax
Weights (64,W), (W,W), (W,10); biases (W,), (W,), (10,); weights ~ N(0, 0.01) (std 0.01), biases 0.
"""
import numpy as np

LAYERS = ("1", "2", "3")


def init_params(width, n_in=64, n_out=10, std=0.01):
    shapes = [(n_in, width), (width, width), (width, n_out)]
    params = {}
    for name, (a, b) in zip(LAYERS, shapes):
        params["W" + name] = np.random.randn(a, b) * std
        params["b" + name] = np.zeros(b)
    return params


def relu(z):
    return np.maximum(z, 0.0)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)  # numerical stability
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def forward(params, X):
    """Return probabilities (batch, 10) and the cache needed for backprop."""
    z1 = X @ params["W1"] + params["b1"]
    a1 = relu(z1)
    z2 = a1 @ params["W2"] + params["b2"]
    a2 = relu(z2)
    z3 = a2 @ params["W3"] + params["b3"]
    p = softmax(z3)
    return p, (X, z1, a1, z2, a2)


def cross_entropy(p, Y, eps=1e-12):
    """Mean cross-entropy over the batch; Y is one-hot."""
    return float(-np.sum(Y * np.log(p + eps)) / Y.shape[0])


def backward(params, p, Y, cache):
    """Gradients of mean cross-entropy w.r.t. every parameter (chain rule).

    dL/dz3 = (p - Y) / B          (softmax + cross-entropy combined)
    dW3 = a2^T dz3, db3 = sum(dz3)
    da2 = dz3 W3^T, dz2 = da2 * 1[z2 > 0]
    dW2 = a1^T dz2, db2 = sum(dz2)
    da1 = dz2 W2^T, dz1 = da1 * 1[z1 > 0]
    dW1 = X^T dz1,  db1 = sum(dz1)
    """
    X, z1, a1, z2, a2 = cache
    B = X.shape[0]
    dz3 = (p - Y) / B
    g = {"W3": a2.T @ dz3, "b3": dz3.sum(axis=0)}
    dz2 = (dz3 @ params["W3"].T) * (z2 > 0)
    g["W2"] = a1.T @ dz2
    g["b2"] = dz2.sum(axis=0)
    dz1 = (dz2 @ params["W2"].T) * (z1 > 0)
    g["W1"] = X.T @ dz1
    g["b1"] = dz1.sum(axis=0)
    return g


def loss_fn(params, X, Y):
    """Same cross-entropy, computed with log-softmax for extra float precision (used by grad check)."""
    a1 = relu(X @ params["W1"] + params["b1"])
    a2 = relu(a1 @ params["W2"] + params["b2"])
    z3 = a2 @ params["W3"] + params["b3"]
    m = z3.max(axis=1, keepdims=True)
    log_p = z3 - m - np.log(np.exp(z3 - m).sum(axis=1, keepdims=True))
    return float(-np.sum(Y * log_p) / Y.shape[0])


def predict(params, X):
    p, _ = forward(params, X)
    return p.argmax(axis=1)


def sgd_step(params, grads, lr):
    for k in params:
        params[k] -= lr * grads[k]
