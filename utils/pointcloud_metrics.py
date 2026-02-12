from __future__ import annotations

"""
GPU-accelerated point cloud metrics using FAISS.

This module computes:
- Pairwise Chamfer distance matrices between sets of point clouds
- MMD (Maximum Mean Discrepancy) with RBF kernel over Chamfer distances
- Sliced Wasserstein Distance (SWD) for point distributions

All heavy computations use FAISS GPU for efficiency.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import ot
import faiss


# =============================================================================
# Utility functions
# =============================================================================

def _as_f32_contig(x: np.ndarray) -> np.ndarray:
    """Convert to contiguous float32 array for FAISS."""
    x = np.asarray(x, dtype=np.float32)
    if not x.flags["C_CONTIGUOUS"]:
        x = np.ascontiguousarray(x)
    return x


def downsample_cloud(x: np.ndarray, m: int, seed: int = 0) -> np.ndarray:
    """
    Deterministic random downsampling of point cloud.
    
    The seed depends ONLY on the cloud identifier, not on whether it's in A or B.
    This ensures consistent downsampling for symmetric computations.
    
    Args:
        x: Point cloud [N, D]
        m: Target number of points
        seed: Random seed (should be base_seed + cloud_index)
    
    Returns:
        Downsampled cloud [min(N, m), D]
    """
    x = _as_f32_contig(x)
    n = x.shape[0]
    if m >= n:
        return x
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=m, replace=False)
    return x[idx]


# =============================================================================
# FAISS GPU utilities
# =============================================================================

@dataclass
class FaissGpuContext:
    """Context for FAISS GPU operations. Create once per validation call."""
    res: faiss.StandardGpuResources
    device: int = 0
    
    @classmethod
    def create(cls, device: int = 0, temp_memory_mb: int = 512) -> "FaissGpuContext":
        """Create GPU context with optional temp memory limit."""
        res = faiss.StandardGpuResources()
        res.setTempMemory(temp_memory_mb * 1024 * 1024)
        return cls(res=res, device=device)


def build_gpu_flat_index(x: np.ndarray, ctx: FaissGpuContext) -> faiss.GpuIndexFlatL2:
    """
    Build exact L2 index on GPU.
    
    Args:
        x: Points [N, D], will be converted to float32
        ctx: GPU context with resources
    
    Returns:
        FAISS GPU index for L2 nearest neighbor search
    """
    x = _as_f32_contig(x)
    d = x.shape[1]
    cfg = faiss.GpuIndexFlatConfig()
    cfg.device = ctx.device
    index = faiss.GpuIndexFlatL2(ctx.res, d, cfg)
    index.add(x)
    return index


def chamfer_distance_gpu(
    a: np.ndarray,
    b: np.ndarray,
    ctx: FaissGpuContext,
    index_b: Optional[faiss.GpuIndexFlatL2] = None,
    index_a: Optional[faiss.GpuIndexFlatL2] = None,
) -> float:
    """
    Symmetric Chamfer distance between two point clouds using FAISS GPU.
    
    Chamfer = mean_{i in a} min_{j in b} ||a_i - b_j||^2 
            + mean_{j in b} min_{i in a} ||b_j - a_i||^2
    
    Args:
        a, b: Point clouds [N, D]
        ctx: GPU context
        index_b: Pre-built index for b (optional, for reuse)
        index_a: Pre-built index for a (optional, for reuse)
    
    Returns:
        Chamfer distance (sum of mean squared distances)
    """
    a = _as_f32_contig(a)
    b = _as_f32_contig(b)

    if index_b is None:
        index_b = build_gpu_flat_index(b, ctx)
    if index_a is None:
        index_a = build_gpu_flat_index(a, ctx)

    # FAISS returns L2^2 distances
    dist_ab, _ = index_b.search(a, 1)  # (Na, 1)
    dist_ba, _ = index_a.search(b, 1)  # (Nb, 1)

    return float(dist_ab.mean() + dist_ba.mean())


# =============================================================================
# Pairwise Chamfer matrix
# =============================================================================

def pairwise_chamfer_matrix_faiss(
    clouds_a: np.ndarray,
    clouds_b: np.ndarray,
    *,
    downsample: int = 1024,
    base_seed: int = 0,
    device: int = 0,
    symmetric: bool = False,
    ctx: Optional[FaissGpuContext] = None,
) -> np.ndarray:
    """
    Compute pairwise Chamfer distance matrix using FAISS GPU.
    
    For symmetric=True (when clouds_a is clouds_b), computes only upper triangle
    and mirrors, ensuring D[i,i] = 0 and D[i,j] = D[j,i].
    
    Args:
        clouds_a: First set of clouds [Na, P, D]
        clouds_b: Second set of clouds [Nb, P, D]
        downsample: Target points per cloud (default 1024)
        base_seed: Base seed for deterministic downsampling
        device: GPU device ID
        symmetric: If True, assumes clouds_a == clouds_b and optimizes
        ctx: Existing GPU context (created if None)
    
    Returns:
        Distance matrix [Na, Nb] of Chamfer distances
    """
    if clouds_a.ndim != 3 or clouds_b.ndim != 3:
        raise ValueError("clouds_a/clouds_b must have shape [Nclouds, Npoints, D].")

    Na, _, D = clouds_a.shape
    Nb, _, D2 = clouds_b.shape
    if D != D2:
        raise ValueError(f"D mismatch: clouds_a has D={D}, clouds_b has D={D2}")

    # Create GPU context once
    if ctx is None:
        ctx = FaissGpuContext.create(device=device)

    # Downsample clouds (deterministic, seed depends only on cloud index)
    # For symmetric case, a and b use the SAME downsampled data
    if symmetric:
        ds = [downsample_cloud(clouds_a[i], downsample, seed=base_seed + i) for i in range(Na)]
        a_ds = ds
        b_ds = ds
    else:
        a_ds = [downsample_cloud(clouds_a[i], downsample, seed=base_seed + i) for i in range(Na)]
        b_ds = [downsample_cloud(clouds_b[j], downsample, seed=base_seed + j) for j in range(Nb)]

    # Build indices for B clouds
    b_indices = [build_gpu_flat_index(b_ds[j], ctx) for j in range(Nb)]

    # Output matrix
    out = np.zeros((Na, Nb), dtype=np.float32)

    if symmetric:
        # Compute only upper triangle (j >= i), mirror to lower
        for i in range(Na):
            idx_a = build_gpu_flat_index(a_ds[i], ctx)
            a = a_ds[i]
            
            for j in range(i + 1, Nb):
                idx_b = b_indices[j]
                b = b_ds[j]
                
                # Chamfer = mean(a->b) + mean(b->a)
                dist_ab, _ = idx_b.search(a, 1)
                dist_ba, _ = idx_a.search(b, 1)
                d = float(dist_ab.mean() + dist_ba.mean())
                
                out[i, j] = d
                out[j, i] = d
        
        # Diagonal is exactly 0 (same cloud)
        np.fill_diagonal(out, 0.0)
    else:
        # Full matrix computation
        for i in range(Na):
            idx_a = build_gpu_flat_index(a_ds[i], ctx)
            a = a_ds[i]
            
            for j in range(Nb):
                idx_b = b_indices[j]
                b = b_ds[j]
                
                dist_ab, _ = idx_b.search(a, 1)
                dist_ba, _ = idx_a.search(b, 1)
                out[i, j] = float(dist_ab.mean() + dist_ba.mean())

    return out


# =============================================================================
# MMD computation
# =============================================================================

def mmd_rbf_from_distance_matrices(
    D_xx: np.ndarray,
    D_yy: np.ndarray,
    D_xy: np.ndarray,
    gamma: float,
) -> float:
    """
    Compute MMD with RBF kernel from precomputed distance matrices.
    
    MMD^2 = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]
    where k(a,b) = exp(-gamma * d(a,b))
    
    Args:
        D_xx: Distance matrix real vs real [N, N]
        D_yy: Distance matrix gen vs gen [M, M]
        D_xy: Distance matrix real vs gen [N, M]
        gamma: RBF kernel bandwidth
    
    Returns:
        MMD value (can be negative due to estimation variance)
    """
    K_xx = np.exp(-gamma * D_xx)
    K_yy = np.exp(-gamma * D_yy)
    K_xy = np.exp(-gamma * D_xy)
    return float(K_xx.mean() + K_yy.mean() - 2.0 * K_xy.mean())


def mmd_rbf_from_chamfer_faiss(
    real_clouds: np.ndarray,
    gen_clouds: np.ndarray,
    gamma: Optional[float] = None,
    *,
    downsample: int = 1024,
    base_seed: int = 0,
    device: int = 0,
    verbose: bool = True,
) -> float:
    """
    Compute MMD with RBF kernel using Chamfer distances on GPU.
    
    This is the main entry point for MMD computation in validation.
    
    Args:
        real_clouds: Real samples [N, P, D]
        gen_clouds: Generated samples [M, P, D]
        gamma: RBF bandwidth. If None, uses 1/median(D_xy)
        downsample: Target points per cloud
        base_seed: Base seed for deterministic downsampling
        device: GPU device ID
        verbose: Log progress
    
    Returns:
        MMD value
    """
    from utils.colorfull_logger import info
    
    N_real, N_gen = len(real_clouds), len(gen_clouds)
    
    # Create context once for all computations
    ctx = FaissGpuContext.create(device=device)
    
    # Compute distance matrices
    # D_xx and D_yy use symmetric=True for efficiency and consistency
    if verbose:
        info(f"Computing D_xx (real×real): {N_real}×{N_real} Chamfer distances...")
    D_xx = pairwise_chamfer_matrix_faiss(
        real_clouds, real_clouds,
        downsample=downsample, base_seed=base_seed, device=device,
        symmetric=True, ctx=ctx,
    )
    
    if verbose:
        info(f"Computing D_yy (gen×gen): {N_gen}×{N_gen} Chamfer distances...")
    D_yy = pairwise_chamfer_matrix_faiss(
        gen_clouds, gen_clouds,
        downsample=downsample, base_seed=base_seed, device=device,
        symmetric=True, ctx=ctx,
    )
    
    if verbose:
        info(f"Computing D_xy (real×gen): {N_real}×{N_gen} Chamfer distances...")
    D_xy = pairwise_chamfer_matrix_faiss(
        real_clouds, gen_clouds,
        downsample=downsample, base_seed=base_seed, device=device,
        symmetric=False, ctx=ctx,
    )
    
    # Auto-select gamma if not provided
    if gamma is None:
        median_dist = float(np.median(D_xy))
        gamma = 1.0 / (median_dist + 1e-8)
    
    if verbose:
        info(f"Computing MMD with gamma={gamma:.4f}...")
    
    return mmd_rbf_from_distance_matrices(D_xx, D_yy, D_xy, gamma)


# =============================================================================
# Legacy API compatibility
# =============================================================================

def pairwise_chamfer_matrix(
    clouds_a: np.ndarray,
    clouds_b: np.ndarray,
    downsample: int = 1024,
    seed: int = 0,
    device: int = 0,
    use_cache: bool = True,
) -> np.ndarray:
    """
    Legacy API wrapper for pairwise_chamfer_matrix_faiss.
    
    Kept for backward compatibility with existing code.
    """
    return pairwise_chamfer_matrix_faiss(
        clouds_a, clouds_b,
        downsample=downsample,
        base_seed=seed,
        device=device,
        symmetric=False,
    )


def mmd_rbf_from_chamfer(
    real_clouds: np.ndarray,
    gen_clouds: np.ndarray,
    gamma: float,
    downsample: int = 1024,
    base_seed: int = 0,
    device: int = 0,
) -> float:
    """
    Legacy API wrapper for mmd_rbf_from_chamfer_faiss.
    """
    return mmd_rbf_from_chamfer_faiss(
        real_clouds, gen_clouds, gamma,
        downsample=downsample,
        base_seed=base_seed,
        device=device,
    )


def chamfer_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Legacy API: Chamfer distance between two clouds.
    Creates GPU context on each call (less efficient for batch use).
    """
    ctx = FaissGpuContext.create(device=0)
    return chamfer_distance_gpu(a, b, ctx)


# =============================================================================
# Sliced Wasserstein Distance (unchanged, uses POT library)
# =============================================================================

def sliced_wasserstein_distance(
    real_points: np.ndarray,
    gen_points: np.ndarray,
    num_projections: int = 256,
    seed: int = 0,
) -> float:
    """
    SWD over point distributions using random projections.
    
    Args:
        real_points: Flattened real points [P, D]
        gen_points: Flattened generated points [P, D]
        num_projections: Number of random 1D projections
        seed: Random seed for projections
    
    Returns:
        Sliced Wasserstein distance
    """
    real_points = _as_f32_contig(real_points)
    gen_points = _as_f32_contig(gen_points)
    
    return float(
        ot.sliced.sliced_wasserstein_distance(
            real_points,
            gen_points,
            n_projections=num_projections,
            seed=seed,
        )
    )


# =============================================================================
# Sanity checks (for testing/debugging)
# =============================================================================

def run_sanity_checks(
    clouds: np.ndarray,
    downsample: int = 1024,
    base_seed: int = 0,
    device: int = 0,
    verbose: bool = True,
) -> Dict[str, bool]:
    """
    Run sanity checks on the Chamfer matrix computation.
    
    Checks:
    1. Symmetry: max|D - D.T| < 1e-5
    2. Diagonal: mean(diag(D)) ~ 0
    3. Determinism: repeated runs produce identical results
    4. GPU availability
    
    Args:
        clouds: Test clouds [N, P, D]
        downsample: Points per cloud
        base_seed: Random seed
        device: GPU device
        verbose: Print results
    
    Returns:
        Dict of check_name -> passed
    """
    results = {}
    
    # GPU availability
    try:
        n_gpus = faiss.get_num_gpus()
        results["gpu_available"] = n_gpus > 0
        if verbose:
            print(f"✓ GPU available: {n_gpus} GPU(s) detected")
    except Exception as e:
        results["gpu_available"] = False
        if verbose:
            print(f"✗ GPU check failed: {e}")
    
    # Compute symmetric matrix
    D = pairwise_chamfer_matrix_faiss(
        clouds, clouds,
        downsample=downsample, base_seed=base_seed, device=device,
        symmetric=True,
    )
    
    # Symmetry check
    sym_error = np.abs(D - D.T).max()
    results["symmetry"] = sym_error < 1e-5
    if verbose:
        status = "✓" if results["symmetry"] else "✗"
        print(f"{status} Symmetry: max|D - D.T| = {sym_error:.2e}")
    
    # Diagonal check
    diag_mean = np.abs(np.diag(D)).mean()
    results["diagonal_zero"] = diag_mean < 1e-8
    if verbose:
        status = "✓" if results["diagonal_zero"] else "✗"
        print(f"{status} Diagonal: mean(|diag|) = {diag_mean:.2e}")
    
    # Determinism check
    D2 = pairwise_chamfer_matrix_faiss(
        clouds, clouds,
        downsample=downsample, base_seed=base_seed, device=device,
        symmetric=True,
    )
    determ_error = np.abs(D - D2).max()
    results["determinism"] = determ_error < 1e-8
    if verbose:
        status = "✓" if results["determinism"] else "✗"
        print(f"{status} Determinism: max|D1 - D2| = {determ_error:.2e}")
    
    return results
