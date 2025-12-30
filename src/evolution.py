from __future__ import annotations
import numpy as np
from copy import deepcopy

def tournament_select(rng: np.random.Generator, population: list[dict], fitness: np.ndarray, k: int = 5) -> dict:
    idxs = rng.integers(0, len(population), size=(k,))
    best = idxs[0]
    for i in idxs[1:]:
        if fitness[i] > fitness[best]:
            best = i
    return population[int(best)]

def crossover(rng: np.random.Generator, parent_a: dict, parent_b: dict) -> dict:
    """
    Uniform crossover per weight element.
    """
    child = deepcopy(parent_a)
    for li in range(len(child["W"])):
        mask = rng.random(child["W"][li].shape) < 0.5
        child["W"][li][mask] = parent_b["W"][li][mask]
        maskb = rng.random(child["b"][li].shape) < 0.5
        child["b"][li][maskb] = parent_b["b"][li][maskb]
    return child

def mutate(rng: np.random.Generator, genome: dict, mut_rate: float = 0.08, mut_scale: float = 0.5) -> dict:
    """
    Add gaussian noise to random parameters.
    """
    for li in range(len(genome["W"])):
        w = genome["W"][li]
        b = genome["b"][li]

        w_mask = rng.random(w.shape) < mut_rate
        b_mask = rng.random(b.shape) < mut_rate

        w[w_mask] += rng.normal(0.0, mut_scale, size=w_mask.sum()).astype(np.float32)
        b[b_mask] += rng.normal(0.0, mut_scale, size=b_mask.sum()).astype(np.float32)

        # mild clipping helps stability
        np.clip(w, -6.0, 6.0, out=w)
        np.clip(b, -6.0, 6.0, out=b)

    return genome