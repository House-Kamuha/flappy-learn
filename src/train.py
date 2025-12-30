from __future__ import annotations
import argparse
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

from .flappy_env import FlappyEnv
from .nn import init_network, forward
from .evolution import tournament_select, crossover, mutate
from .genome_io import save_genome

def evaluate_genome(genome: dict, seeds: list[int], max_steps: int = 6000) -> float:
    """
    Fitness = average over seeds of (frames_alive + pipes_passed*500)
    """
    scores = []
    for s in seeds:
        env = FlappyEnv(max_frames=max_steps)
        obs = env.reset(seed=s)

        done = False
        while not done:
            out = forward(genome, obs)
            flap = out > 0.0
            obs, done, info = env.step(flap)

        fitness = float(info["frames"] + info["score"] * 500)
        scores.append(fitness)

    return float(np.mean(scores))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gens", type=int, default=120)
    parser.add_argument("--pop", type=int, default=200)
    parser.add_argument("--elite_frac", type=float, default=0.10)
    parser.add_argument("--tournament_k", type=int, default=6)
    parser.add_argument("--mut_rate", type=float, default=0.08)
    parser.add_argument("--mut_scale", type=float, default=0.5)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--outdir", type=str, default="artifacts")
    parser.add_argument("--results", type=str, default="results")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # 6 inputs -> 8 hidden -> 1 output
    layer_sizes = [6, 8, 1]

    population = [init_network(rng, layer_sizes) for _ in range(args.pop)]
    elite_n = max(2, int(args.pop * args.elite_frac))

    outdir = Path(args.outdir)
    resdir = Path(args.results)
    outdir.mkdir(parents=True, exist_ok=True)
    resdir.mkdir(parents=True, exist_ok=True)

    csv_path = resdir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["generation", "max_fitness", "mean_fitness"])

    best_fitness_ever = -1e9
    best_genome_ever = None
    history_max = []
    history_mean = []

    for gen in range(args.gens):
        # fixed evaluation seeds for fairness (per generation)
        seeds = [1000 * gen + i for i in range(args.episodes)]

        fitness = np.zeros(args.pop, dtype=np.float32)
        for i in range(args.pop):
            fitness[i] = evaluate_genome(population[i], seeds=seeds)

        max_fit = float(fitness.max())
        mean_fit = float(fitness.mean())
        history_max.append(max_fit)
        history_mean.append(mean_fit)

        # record
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([gen, max_fit, mean_fit])

        # track best ever
        best_idx = int(fitness.argmax())
        if max_fit > best_fitness_ever:
            best_fitness_ever = max_fit
            best_genome_ever = population[best_idx]
            save_genome(outdir / "best_genome.npz", best_genome_ever)

        # select elites
        elite_idxs = np.argsort(fitness)[-elite_n:]
        elites = [population[int(i)] for i in elite_idxs]

        # next generation
        next_pop = []
        # keep elites (elitism)
        for e in elites:
            # deep copy arrays so we don't mutate elites in-place
            copied = {
                "layer_sizes": e["layer_sizes"],
                "W": [w.copy() for w in e["W"]],
                "b": [b.copy() for b in e["b"]],
            }
            next_pop.append(copied)

        # fill rest with children
        while len(next_pop) < args.pop:
            p1 = tournament_select(rng, population, fitness, k=args.tournament_k)
            p2 = tournament_select(rng, population, fitness, k=args.tournament_k)
            child = crossover(rng, p1, p2)
            child = mutate(rng, child, mut_rate=args.mut_rate, mut_scale=args.mut_scale)
            next_pop.append(child)

        population = next_pop

        print(f"Gen {gen:03d} | max={max_fit:8.1f} | mean={mean_fit:8.1f} | best_ever={best_fitness_ever:8.1f}")

    # plot training curve
    plt.figure()
    plt.plot(history_max, label="max_fitness")
    plt.plot(history_mean, label="mean_fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.tight_layout()
    plt.savefig(resdir / "training_curve.png", dpi=160)
    print(f"\nSaved best genome to: {outdir/'best_genome.npz'}")
    print(f"Saved results CSV to: {csv_path}")
    print(f"Saved training curve to: {resdir/'training_curve.png'}")

if __name__ == "__main__":
    main()