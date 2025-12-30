from __future__ import annotations
import numpy as np
from pathlib import Path

def save_genome(path: str | Path, genome: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays = {"layer_sizes": np.array(genome["layer_sizes"], dtype=np.int32)}
    for i, w in enumerate(genome["W"]):
        arrays[f"W{i}"] = w
    for i, b in enumerate(genome["b"]):
        arrays[f"b{i}"] = b

    np.savez_compressed(path, **arrays)

def load_genome(path: str | Path) -> dict:
    data = np.load(Path(path), allow_pickle=False)
    layer_sizes = data["layer_sizes"].astype(int).tolist()

    W = []
    b = []
    i = 0
    while f"W{i}" in data:
        W.append(data[f"W{i}"].astype(np.float32))
        i += 1

    i = 0
    while f"b{i}" in data:
        b.append(data[f"b{i}"].astype(np.float32))
        i += 1

    return {"W": W, "b": b, "layer_sizes": layer_sizes}