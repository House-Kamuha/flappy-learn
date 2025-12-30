from __future__ import annotations
import numpy as np

def init_network(rng: np.random.Generator, layer_sizes: list[int], weight_scale: float = 0.7):
    """
    Returns a genome dict: {"W": [...], "b": [...]}
    """
    W = []
    b = []
    for i in range(len(layer_sizes) - 1):
        in_dim = layer_sizes[i]
        out_dim = layer_sizes[i + 1]
        W.append(rng.normal(0.0, weight_scale, size=(out_dim, in_dim)).astype(np.float32))
        b.append(rng.normal(0.0, weight_scale, size=(out_dim,)).astype(np.float32))
    return {"W": W, "b": b, "layer_sizes": layer_sizes}

def forward(genome, x: np.ndarray) -> float:
    """
    x: shape (in_dim,)
    returns scalar output in [-1, 1]
    """
    a = x.astype(np.float32)
    for i in range(len(genome["W"])):
        z = genome["W"][i] @ a + genome["b"][i]
        # tanh hidden layers, tanh output
        a = np.tanh(z)
    # final layer is also tanh; return single value
    return float(a[0])