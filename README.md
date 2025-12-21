# flappy-learn 🧬🐦
**From-scratch neuroevolution** that trains a *population* of tiny neural networks to play a Flappy Bird–style game.

> what goes up must come DOWN.

![Demo](assets/demo.gif)

---

## What this project is
This repo simulates a Flappy Bird–style environment (pipes + gravity + collisions) and evolves a population of agents over generations.

Each agent is a small neural network. Agents that survive longer and pass more pipes get higher fitness, and their “brains” are selected + mutated to create the next generation.

**No RL libraries, no NEAT library** — the learning loop (selection / crossover / mutation) is implemented from scratch.

---

## Quickstart

### Install
```bash
pip install -r requirements.txt
