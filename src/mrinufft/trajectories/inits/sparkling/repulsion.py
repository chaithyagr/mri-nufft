"""Repulsion force computation for sparkling trajectories.

Supports CPU (numpy) and GPU (cupy) backends, with optional pykeops GPU acceleration.
The repulsion force enforces uniform sampling density by pushing points apart.

This module is self-contained and does not depend on the full mrinufft package.
"""

# Lazy import of get_array_module to avoid breaking circular imports
_get_array_module_cache = {}


def _get_array_module():
    """Get the array module, lazily imported from mrinufft."""
    if "xp" not in _get_array_module_cache:
        try:
            from mrinufft._array_compat import get_array_module

            _get_array_module_cache["xp"] = get_array_module
        except ImportError:
            import numpy as np

            _get_array_module_cache["xp"] = np
    return _get_array_module_cache["xp"]


def compute_repulsion_force(points, epsilon=0.01, use_pykeops=False, time=None):
    """Compute the repulsion force on each point.

    Computes the repulsion gradient from the FMM potential::

        F[i, d] = (1/N) * sum_j (x_i[d] - x_j[d]) / ||x_i - x_j||

    Parameters
    ----------
    points : NDArray
        Trajectory points of shape (N, D).
    epsilon : float
        Smoothing parameter for the kernel.
    use_pykeops : bool
        Whether to use pykeops for GPU-accelerated computation.
        When ``False``, uses direct numpy/cupy computation (O(N^2)).
    time : NDArray | None
        Temporal weights for temporal repulsion.

    Returns
    -------
    NDArray
        Repulsion force of shape (N, D).

    Raises
    ------
    ImportError
        If pykeops is requested but not available.
    TypeError
        If pykeops is requested on a non-GPU array.
    """
    xp = _get_array_module()
    eps = epsilon ** 2

    if use_pykeops:
        return _compute_repulsion_pykeops(points, eps, time)
    else:
        return _compute_repulsion_direct(points, eps)


def _compute_repulsion_direct(points, eps):
    """Direct O(N^2) computation using numpy or cupy."""
    xp = _get_array_module()
    N, D = points.shape

    diff = points[:, None, :] - points[None, :, :]
    dist = xp.sqrt((diff ** 2).sum(-1) + eps)
    inv_dist = 1.0 / dist

    Q_weighted = (inv_dist[:, :, None] * points[None, :, :]).sum(1)
    Q_scalar = inv_dist.sum(1)
    force = (points * Q_scalar[:, None] - Q_weighted) / N

    return force


def _compute_repulsion_pykeops(points, eps, time):
    """GPU-accelerated computation using pykeops."""
    xp = _get_array_module()
    array_name = getattr(xp, "__name__", "")
    N, D = points.shape

    if array_name == "cupy":
        try:
            from pykeops.numpy import LazyTensor
        except ImportError:
            raise ImportError(
                "pykeops is required for GPU-accelerated repulsion. "
                "Install with: pip install pykeops"
            )
        return _pykeops_impl(LazyTensor, eps, N, D, points, time)
    elif array_name == "torch":
        try:
            from pykeops.torch import LazyTensor
        except ImportError:
            raise ImportError(
                "pykeops is required for GPU-accelerated repulsion. "
                "Install with: pip install pykeops"
            )
        return _pykeops_impl(LazyTensor, eps, N, D, points, time)
    else:
        raise TypeError("pykeops repulsion requires GPU arrays (cupy/torch)")


def _pykeops_impl(LazyTensor, eps, N, D, weights, time):
    """Core pykeops repulsion implementation."""
    xp = _get_array_module()
    # Build w = [1, x_0, x_1, ..., x_{D-1}]
    ones = xp.ones((N, 1), dtype=weights.dtype)
    w = xp.hstack([ones, weights])

    x_i = LazyTensor(w[:, None, :])  # (N, 1, D+1)
    y_j = LazyTensor(w[None, :, :])  # (1, N, D+1)

    # Spatial difference: skip first column (weight = 1)
    spatial_diff = x_i[1:] - y_j[1:]
    inv_dist = ((spatial_diff ** 2).sum(-1) + eps).rsqrt()  # (N, N)

    if time is not None:
        t_i = LazyTensor(time[:, None, None])
        t_j = LazyTensor(time[None, :, None])
        inv_dist = inv_dist * (LazyTensor(t_i - t_j).exp())

    src_spatial = y_j[0, :, 1:]  # (N, D)
    Q_weighted = (inv_dist[:, :, None] * src_spatial[None, :, :]).sum(1)  # (N, D)
    Q_scalar = inv_dist.sum(1)  # (N,)

    force = (weights * Q_scalar[:, None] - Q_weighted) / N

    return force
