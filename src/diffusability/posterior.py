"""Gaussian posterior mixtures and their exact linear-flow marginal velocity."""
from __future__ import annotations

import hashlib
import math

import numpy as np
import torch
from scipy.optimize import brentq


def stream_seed(seed: int, namespace: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{namespace}".encode()).digest()[:8], "little") % (2**63 - 1)


def generator(seed: int, namespace: str, device: torch.device | str = "cpu") -> torch.Generator:
    return torch.Generator(device=device).manual_seed(stream_seed(seed, namespace))


def covariance_profiles(dim: int, trace: float, logdet: float, multiplicities: list[int]) -> dict[str, np.ndarray]:
    if len(multiplicities) != 3 or len(set(multiplicities)) != 3:
        raise ValueError("Exactly three distinct multiplicities are required.")
    if trace <= 0 or logdet >= dim * math.log(trace / dim):
        raise ValueError("Non-isotropic profiles require logdet < D log(trace/D).")
    profiles = []
    for m in multiplicities:
        if not 0 < m < dim:
            raise ValueError("Multiplicity must lie strictly between zero and D.")
        def equation(a: float) -> float:
            b = (trace - m * a) / (dim - m)
            return m * math.log(a) + (dim - m) * math.log(b) - logdet
        a = brentq(equation, 1e-14, trace / dim, xtol=1e-14)
        b = (trace - m * a) / (dim - m)
        s = np.r_[np.full(m, a), np.full(dim - m, b)]
        profiles.append(s)
    profiles.sort(key=lambda s: np.var(np.log(s)))
    perm = np.random.default_rng(42).permutation(dim)
    result = {name: s[perm] for name, s in zip(["low", "mid", "high"], profiles)}
    for s in result.values():
        if not np.isclose(s.sum(), trace, atol=1e-10, rtol=0) or not np.isclose(np.log(s).sum(), logdet, atol=1e-9, rtol=0):
            raise ValueError("Covariance invariant failure.")
    return result


def make_centers(dim: int, components: int, radius: float, seed: int) -> torch.Tensor:
    mu = torch.randn(components, dim, generator=generator(seed, "centers"), dtype=torch.float64)
    mu -= mu.mean(0)
    return mu * (radius / mu.square().sum(1).mean().sqrt())


def draw(mu: torch.Tensor, s: torch.Tensor, n: int, rng: torch.Generator) -> torch.Tensor:
    labels = torch.randint(len(mu), (n,), device=mu.device, generator=rng)
    eps = torch.randn(n, mu.shape[1], device=mu.device, dtype=mu.dtype, generator=rng)
    return mu[labels] + eps * s.sqrt()


def path_batch(mu: torch.Tensor, s: torch.Tensor, n: int, rng: torch.Generator):
    target = draw(mu, s, n, rng)
    source = torch.randn(target.shape, device=mu.device, dtype=mu.dtype, generator=rng)
    t = torch.rand(n, 1, device=mu.device, dtype=mu.dtype, generator=rng)
    return (1 - t) * source + t * target, t, target - source


def log_density(x: torch.Tensor, t: torch.Tensor, mu: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    c = (1 - t).square() + t.square() * s
    delta = x[:, None, :] - t[:, None, :] * mu[None, :, :]
    logits = -0.5 * (delta.square() / c[:, None, :]).sum(-1)
    return torch.logsumexp(logits, dim=1) - math.log(len(mu)) - 0.5 * (c.log().sum(-1) + x.shape[1] * math.log(2 * math.pi))


def exact_velocity(x: torch.Tensor, t: torch.Tensor, mu: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    c = (1 - t).square() + t.square() * s
    delta = x[:, None, :] - t[:, None, :] * mu[None, :, :]
    logits = -0.5 * (delta.square() / c[:, None, :]).sum(-1)
    mean = torch.softmax(logits, dim=1) @ mu
    return mean + (t * s - (1 - t)) / c * (x - t * mean)


@torch.no_grad()
def heun(velocity, noise: torch.Tensor, steps: int) -> torch.Tensor:
    x = noise.clone()
    dt = 1.0 / steps
    for i in range(steps):
        t = x.new_full((len(x), 1), i * dt)
        first = velocity(x, t)
        second = velocity(x + dt * first, t + dt)
        x = x + 0.5 * dt * (first + second)
    return x


def sliced_w2(x: torch.Tensor, y: torch.Tensor, directions: torch.Tensor) -> float:
    """Root mean squared 1D W2 over fixed unit directions; equal sample counts."""
    if x.shape != y.shape:
        raise ValueError("SWD requires matching sample count and dimension.")
    xp = (x @ directions).sort(dim=0).values
    yp = (y @ directions).sort(dim=0).values
    return float((xp - yp).square().mean().sqrt())
