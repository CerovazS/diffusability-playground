# Diffusability Playground

<p>
  <strong>Research goal:</strong> study diffusability of latent spaces by starting from synthetic datasets with controllable geometry, evaluating them with DiT models, and later moving to real audio and vision datasets.
</p>

<p>
  <span style="color:#f1c40f"><strong>INFO:</strong> This repository is under active construction.</span>
</p>

## Quick Context
- **Env/deps:** use Astral `uv` (`uv add` only, no `pip install`).
- **Training/config:** Hydra-driven experiments and configs.
- **Docs:** `https://docs.astral.sh/uv/` and `https://hydra.cc/docs/intro/`.
- **Maintenance:** after each operation, update this README to reflect new changes or fix outdated info.

## Synthetic Point Clouds (brief)
`synthetic_pointclouds.py` implements a deterministic, Hydra-friendly dataset generator.  
Each sample is a point cloud `x ∈ R^{N×D}` with label `y`, produced by selecting a class and a mixture component, sampling intrinsic coordinates `z` from a configurable tail distribution, mapping them into ambient space, and adding thickness noise.  
It supports multiple geometric families (affine subspaces, sine-warped subspaces, and MoG) with per-class controls for intrinsic dimension, ambient dimension, number of modes, separation, anisotropy, curvature, and tail heaviness.
