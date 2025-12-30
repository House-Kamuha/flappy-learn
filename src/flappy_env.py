from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np

@dataclass
class Pipe:
    x: float
    gap_y: float  # center of gap
    passed: bool = False

class FlappyEnv:
    """
    Simple Flappy Bird-like environment.
    Headless by default; pygame rendering is done in render_best.py
    """
    def __init__(
        self,
        width: int = 288,
        height: int = 512,
        pipe_width: int = 52,
        gap_size: int = 140,
        pipe_speed: float = 2.8,
        gravity: float = 0.55,
        flap_impulse: float = -9.0,
        bird_x: int = 60,
        bird_radius: int = 12,
        pipe_spawn_frames: int = 90,
        max_frames: int = 6000,
    ):
        self.width = width
        self.height = height
        self.pipe_width = pipe_width
        self.gap_size = gap_size
        self.pipe_speed = pipe_speed
        self.gravity = gravity
        self.flap_impulse = flap_impulse
        self.bird_x = bird_x
        self.bird_radius = bird_radius
        self.pipe_spawn_frames = pipe_spawn_frames
        self.max_frames = max_frames

        self.rng = np.random.default_rng(0)
        self.reset()

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.bird_y = self.height * 0.45
        self.bird_vy = 0.0
        self.frames = 0
        self.score = 0
        self.alive = True

        self.pipes: list[Pipe] = []
        # spawn first pipe a bit ahead
        self._spawn_pipe(x=self.width + 100)

        return self._get_obs()

    def step(self, flap: bool) -> tuple[np.ndarray, bool, dict]:
        """
        flap=True applies impulse this frame.
        returns: observation, done, info
        """
        if not self.alive:
            return self._get_obs(), True, {"score": self.score, "frames": self.frames}

        self.frames += 1

        if flap:
            self.bird_vy = self.flap_impulse

        # physics
        self.bird_vy += self.gravity
        self.bird_y += self.bird_vy

        # move pipes
        for p in self.pipes:
            p.x -= self.pipe_speed

        # remove offscreen pipes
        self.pipes = [p for p in self.pipes if p.x + self.pipe_width > -20]

        # spawn pipes on interval
        if self.frames % self.pipe_spawn_frames == 0:
            self._spawn_pipe(x=self.width + 20)

        # scoring: when bird passes pipe's right edge
        for p in self.pipes:
            if not p.passed and (p.x + self.pipe_width) < self.bird_x:
                p.passed = True
                self.score += 1

        # collisions
        if self._collided():
            self.alive = False

        done = (not self.alive) or (self.frames >= self.max_frames)
        return self._get_obs(), done, {"score": self.score, "frames": self.frames}

    def _spawn_pipe(self, x: float) -> None:
        margin = 60
        min_center = margin + self.gap_size / 2
        max_center = self.height - margin - self.gap_size / 2
        gap_y = float(self.rng.uniform(min_center, max_center))
        self.pipes.append(Pipe(x=x, gap_y=gap_y))

    def _next_pipe(self) -> Pipe:
        # next pipe ahead of bird
        ahead = [p for p in self.pipes if p.x + self.pipe_width >= self.bird_x - 5]
        if ahead:
            return min(ahead, key=lambda p: p.x)
        # fallback
        return self.pipes[0]

    def _get_obs(self) -> np.ndarray:
        """
        Observation vector (small + stable):
        - bird_y normalized
        - bird_vy normalized
        - dx to next pipe normalized
        - gap_center normalized
        - gap_top normalized
        - gap_bottom normalized
        """
        p = self._next_pipe()
        dx = (p.x + self.pipe_width - self.bird_x)

        gap_top = p.gap_y - self.gap_size / 2
        gap_bottom = p.gap_y + self.gap_size / 2

        obs = np.array([
            self.bird_y / self.height,
            np.clip(self.bird_vy / 12.0, -1.0, 1.0),
            np.clip(dx / self.width, -1.0, 1.0),
            p.gap_y / self.height,
            gap_top / self.height,
            gap_bottom / self.height,
        ], dtype=np.float32)
        return obs

    def _collided(self) -> bool:
        # floor / ceiling
        if self.bird_y - self.bird_radius <= 0:
            return True
        if self.bird_y + self.bird_radius >= self.height:
            return True

        p = self._next_pipe()
        gap_top = p.gap_y - self.gap_size / 2
        gap_bottom = p.gap_y + self.gap_size / 2

        # bird circle approx vs pipe rects (cheap check)
        bird_left = self.bird_x - self.bird_radius
        bird_right = self.bird_x + self.bird_radius
        bird_top = self.bird_y - self.bird_radius
        bird_bottom = self.bird_y + self.bird_radius

        pipe_left = p.x
        pipe_right = p.x + self.pipe_width

        # horizontal overlap?
        if bird_right < pipe_left or bird_left > pipe_right:
            return False

        # if overlapping pipe columns, must be within gap vertically
        if bird_top < gap_top or bird_bottom > gap_bottom:
            return True

        return False