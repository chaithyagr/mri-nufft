"""Density-based attraction force for sparkling trajectories.

The attraction force pushes trajectory points toward undersampled regions
of the target density grid.  The density is always expected as a flat array
passed by the user (no grid generation here).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def compute_attraction_force(
    points: NDArray,
    target_density: NDArray,
    grid_centers: NDArray,
    grid_dx: float | NDArray,
    k_max: float = 0.5,
) -> NDArray:
    """Compute the attraction force towards the target density grid.

    Parameters
    ----------
    points : NDArray
        Trajectory points of shape ``(N, D)``.
    target_density : NDArray
        Target density on the grid (flat), **not** normalised.
    grid_centers : NDArray
        Grid cell centers of shape ``(Ng, D)``.
    grid_dx : float or array of floats
        Grid spacing per dimension (in normalised k-space units, i.e.
        ``k_max * 2 / grid_size``).  A scalar or shape *(D)* array is accepted.
    k_max : float
        Maximum coordinate in the normalised grid (used to scale spacing).

    Returns
    -------
    NDArray
        Attraction gradient force of shape ``(N, D)``.
    """
    N, D = points.shape

    # Normalise grid spacing to scalar
    dx = float(np.mean(np.abs(grid_dx))) if hasattr(grid_dx, "__len__") else abs(float(grid_dx))

    # Find closest grid cell for each point
    diff = points[:, None, :] - grid_centers[None, :, :]  # (N, Ng, D)
    dist_sq = np.sum(diff ** 2, axis=-1)  # (N, Ng)
    closest = np.argmin(dist_sq, axis=1)  # (N,)

    # Normalise density to get probability mass per cell
    norm_dens = target_density / np.sum(target_density)

    # Current occupancy: which cells are occupied
    current_occupancy = np.bincount(closest, minlength=target_density.size)
    mask = (current_occupancy > 0).astype(np.float64)

    target = target_density.ravel() if target_density.ndim > 1 else target_density
    # Target occupancy based on density mass
    target_occupancy = np.clip(np.round(norm_dens * np.sum(mask)), 0, None)

    error = target_occupancy - mask

    # Compute force
    force = np.zeros((N, D), dtype=np.float64)
    for d in range(D):
        disp = grid_centers[closest, d] - points[:, d]  # (N,)
        force[:, d] = error[closest] * disp

    return force
