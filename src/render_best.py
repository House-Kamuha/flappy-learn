from __future__ import annotations
import argparse
import pygame
import numpy as np

from .flappy_env import FlappyEnv
from .nn import forward
from .genome_io import load_genome

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="Path to best_genome.npz")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=60)
    args = parser.parse_args()

    genome = load_genome(args.model)
    env = FlappyEnv()
    obs = env.reset(seed=args.seed)

    pygame.init()
    screen = pygame.display.set_mode((env.width, env.height))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 18)

    running = True
    while running:
        clock.tick(args.fps)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        out = forward(genome, obs)
        flap = out > 0.0
        obs, done, info = env.step(flap)

        # draw
        screen.fill((20, 20, 30))

        # pipes
        for p in env.pipes:
            gap_top = p.gap_y - env.gap_size / 2
            gap_bottom = p.gap_y + env.gap_size / 2
            pipe_rect_top = pygame.Rect(int(p.x), 0, env.pipe_width, int(gap_top))
            pipe_rect_bot = pygame.Rect(int(p.x), int(gap_bottom), env.pipe_width, env.height - int(gap_bottom))
            pygame.draw.rect(screen, (60, 200, 120), pipe_rect_top)
            pygame.draw.rect(screen, (60, 200, 120), pipe_rect_bot)

        # bird
        pygame.draw.circle(screen, (220, 220, 80), (env.bird_x, int(env.bird_y)), env.bird_radius)

        # HUD
        text = font.render(f"Score: {info['score']}  Frames: {info['frames']}  flap:{int(flap)}", True, (230, 230, 230))
        screen.blit(text, (10, 10))

        pygame.display.flip()

        if done:
            # restart with same seed to watch consistent behavior
            obs = env.reset(seed=args.seed)

    pygame.quit()

if __name__ == "__main__":
    main()