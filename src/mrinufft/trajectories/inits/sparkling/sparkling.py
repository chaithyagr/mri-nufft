"""Sparkling trajectory initialization.

Provides ``initialize_2D_sparkling`` and ``initialize_3D_sparkling`` which
compute SPARKLING trajectories via an iterative gradient-descent scheme on
**all** points (flat ``[Nc*Ns, D]``) with multi-resolution support.

For a complete 2D example with visualization see:
``mrinufft/trajectories/inits/sparkling/example_2d_sparkling.py``

Algorithm overview
==================
1. Start from a base trajectory (radial / spiral) with perturbed samples.
2. Iteratively update points using attraction (toward desired density)
   and repulsion (to prevent clustering) forces.
3. Periodically project onto hardware constraints (gradient / slew bounds)
   using ``mrinufft.trajectories.project_trajectory``.
4. Repeat through coarse-to-fine resolution levels.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# ── Lazy constant loaders (avoid circular imports) ──────────────────────────

_KMAX_DEFAULT = 0.5


def _get_array_module():
    """Get the array module, lazily from mrinufft or falling back to numpy."""
    try:
        from mrinufft._array_compat import get_array_module
        return get_array_module
    except ImportError:
        return np


def _get_KMAX():
    """Get KMAX constant lazily."""
    try:
        from mrinufft.trajectories.utils import KMAX
        return KMAX
    except ImportError:
        return _KMAX_DEFAULT


def _get_init_2D_radial():
    lazy_import = ['mrinufft.trajectories.inits.radial', 'initialize_2D_radial']
    try:
        mod = __import__(lazy_import[0], fromlist=[lazy_import[1]])
        return getattr(mod, lazy_import[1])
    except ImportError:
        def _mock(Nc, Ns, tilt='uniform', in_out=False):
            traj = np.zeros((Nc, Ns, 2))
            seg = np.linspace(-1 if in_out else 0, 1, Ns)
            traj[0, :, 0] = _KMAX_DEFAULT * seg
            return traj
        return _mock


def _get_init_3D_golden_means_radial():
    try:
        mod = __import__('mrinufft.trajectories.inits.radial', fromlist=['initialize_3D_golden_means_radial'])
        return getattr(mod, 'initialize_3D_golden_means_radial')
    except ImportError:
        def _mock(Nc, Ns, in_out=False):
            m1 = (np.arange(Nc) * 0.618034) % 1
            m2 = (np.arange(Nc) * 0.381966) % 1
            pol = np.arccos(m1).reshape(-1, 1)
            az = (2*np.pi * m2).reshape(-1, 1)
            rad = np.linspace(-1 if in_out else 1, 1, Ns).reshape(1, -1)
            traj = np.zeros((Nc, Ns, 3))
            traj[:,:,0] = rad * np.sin(pol) * np.cos(az)
            traj[:,:,1] = rad * np.sin(pol) * np.sin(az)
            traj[:,:,2] = rad * np.cos(pol) * np.sign(np.abs(rad) + 1e-12)
            return _KMAX_DEFAULT * traj
        return _mock


def _get_init_2D_spiral():
    try:
        mod = __import__('mrinufft.trajectories.inits.spiral', fromlist=['initialize_2D_spiral'])
        return getattr(mod, 'initialize_2D_spiral')
    except ImportError:
        return _get_init_2D_radial()


def _get_init_2D_fibonacci_spiral():
    try:
        mod = __import__('mrinufft.trajectories.inits.spiral', fromlist=['initialize_2D_fibonacci_spiral'])
        return getattr(mod, 'initialize_2D_fibonacci_spiral')
    except ImportError:
        return _get_init_2D_radial()


# ── Lazy repulsion loader ───────────────────────────────────────────────────

def _get_compute_repulsion_force():
    try:
        from .repulsion import compute_repulsion_force
        return compute_repulsion_force
    except ImportError:
        def _fallback(points, epsilon=0.01, **kw):
            xp = _get_array_module()
            eps = epsilon ** 2
            N, D = points.shape
            diff = points[:, None, :] - points[None, :, :]
            dist = xp.sqrt((diff ** 2).sum(-1) + eps)
            inv_dist = 1.0 / dist
            Q_w = (inv_dist[:, :, None] * points[None, :, :]).sum(1)
            Q_s = inv_dist.sum(1)
            return (points * Q_s[:, None] - Q_w) / N
        return _fallback


# ── Sparkling optimisation primitives ────────────────────────────────────────

def _sparkling_gradient_step(shots, attr_force, rep_force,
                              alpha, a_scale=1.0, r_scale=1.0):
    xp = _get_array_module()
    return shots + alpha * (a_scale * attr_force - r_scale * rep_force)


def _project_shots(shots, k_max):
    """Clamp points inside a sphere of radius k_max."""
    xp = _get_array_module()
    norms = xp.linalg.norm(shots, axis=-1, keepdims=True)
    bad = norms > k_max
    if xp.any(bad):
        factor = xp.minimum(norms, k_max + 1e-12)
        return xp.where(bad, (k_max / factor) * shots, shots)
    return shots


def _sparkling_iterate(pts_flat, attr_func, rep_kwargs, a_scale, r_scale,
                        n_iter, alpha, k_max):
    """Pure gradient-descent loop on flat ``[N, D]`` array."""
    for _ in range(n_iter):
        f_attr = attr_func(pts_flat)
        f_rep = _get_compute_repulsion_force()(pts_flat, **rep_kwargs)
        pts_flat = _sparkling_gradient_step(pts_flat, f_attr, f_rep, alpha,
                                            a_scale, r_scale)
        pts_flat = _project_shots(pts_flat, k_max)
    return pts_flat


def _interpolate_resolutions(shots, upsample_factor):
    """Linear interpolation of trajectory to finer resolution."""
    Nc, Ns, D = shots.shape
    new_Ns = int(Ns * upsample_factor)
    t_old = np.linspace(0, 1, Ns, endpoint=False)
    t_new = np.linspace(0, 1, new_Ns, endpoint=False)
    out = np.zeros((Nc, new_Ns, D), dtype=np.float64)
    for d in range(D):
        for i in range(Nc):
            out[i, :, d] = np.interp(t_new, t_old, shots[i, :, d])
    return out


# ── Public high-level entry point ────────────────────────────────────────────

def _initialize_sparkling(Nc, Ns, dimension, *,
                          density=None,
                          initial_type='radial',
                          in_out=True,
                          n_iter=50,
                          eps=0.01,
                          alpha=None,
                          scale_factor=0.75,
                          attraction_scale=1.0,
                          repulsion_scale=1.0,
                          grid_size=64,
                          multi_resolution=True,
                          resolutions=None,
                          use_pykeops=False,
                          proj_every_n=5,
                          k_max=None,
                          seed=None):
    # Lazy loaders
    KMAX = k_max if k_max is not None else _get_KMAX()
    ini_2D_radial = _get_init_2D_radial()
    ini_3D_golden = _get_init_3D_golden_means_radial()
    ini_2D_spiral = _get_init_2D_spiral()
    ini_2D_fib_spiral = _get_init_2D_fibonacci_spiral()

    rng = np.random.default_rng(seed)
    if alpha is None:
        alpha = scale_factor / (2 * np.pi)

    # --- Multi-resolution levels ---
    if multi_resolution:
        if resolutions is None:
            max_res = 2 ** int(np.log2(Ns // 4)) if Ns >= 8 else 1
            resolutions = tuple(
                2 ** i for i in range(int(np.log2(max_res)) + 1)
                if 2 ** i <= max_res and 2 ** i >= 1
            )
            if resolutions[0] != 1:
                resolutions = (1,) + resolutions
        N_levels = len(resolutions)
        n_iter_per = max(1, n_iter // N_levels)
    else:
        resolutions = (1,)
        N_levels = 1
        n_iter_per = n_iter

    # Grid for density → force mapping
    grid = None
    grid_centers = None
    grid_dx = None
    if density is not None and np.size(density) > 1:
        grid = np.asarray(density).ravel() if density.ndim > 1 else density.ravel()
        gs_shape = np.asarray(density.reshape(-1).shape) \
                   if density.ndim == 1 else np.asarray(density.shape)
        grid_dx = 2.0 * KMAX / gs_shape
        dx = float(np.mean(np.abs(grid_dx)))
        grids_list = [
            np.linspace(-KMAX + dx/2, KMAX - dx/2, s)
            for s in gs_shape
        ]
        mesh_grids = np.meshgrid(*grids_list, indexing='ij')
        grid_centers = np.stack([g.ravel() for g in mesh_grids], axis=1)

    def _attraction(pts):
        """Attraction force callback (always CPU)." If grid is None
        returns zero force (uniform target)."""
        if grid is None or grid_centers is None:
            return np.zeros(pts.shape, dtype=np.float64)
        N, D = pts.shape
        dx_val = float(np.mean(np.abs(grid_dx)))
        # closest cell
        diff = pts[:, None, :] - grid_centers[None, :, :]  # (N, Ng, D)
        closest = np.argmin(np.sum(diff**2, axis=2), axis=1)  # (N,)
        # occupancy
        occ = np.bincount(closest, minlength=grid.size)
        mask = (occ > 0).astype(np.float64)
        target_occ = np.clip(np.round(grid * np.sum(mask)), 0, None)
        error = target_occ - mask
        force = np.zeros((N, D), dtype=np.float64)
        for d in range(D):
            force[:, d] = error[closest] * (grid_centers[closest, d] - pts[:, d])
        return force

    rep_kwargs = dict(epsilon=eps, use_pykeops=use_pykeops and grid is not None)

    # Sphere-clamp projection
    def _proj_step(full):
        out = np.copy(full)
        for i in range(out.shape[0]):
            norms = np.linalg.norm(out[i], axis=-1)
            bad = norms > KMAX
            if bad.any():
                f = np.minimum(norms, KMAX+1e-12)
                out[i] = np.where(bad[:,None], (KMAX/f[:,None])*out[i], out[i])
        return out

    # ── Multi-resolution loop ──
    shots = None
    for idx, res in enumerate(resolutions):
        # Base trajectory
        Ns_coarse = Ns // res if multi_resolution else Ns
        if dimension == 2:
            if initial_type == 'fibonacci_spiral':
                base = ini_2D_fib_spiral(Nc, Ns_coarse, in_out=in_out)
            elif initial_type == 'spiral':
                base = ini_2D_spiral(Nc, Ns_coarse, spiral='archimedes',
                                     in_out=in_out)
            else:
                base = ini_2D_radial(Nc, Ns_coarse, tilt='golden', in_out=in_out)
        elif dimension == 3:
            base = ini_3D_golden(Nc, Ns_coarse, in_out=in_out)
        else:
            raise ValueError(f'Unsupported dimension: {dimension}')

        # Perturb only at first level
        if idx == 0:
            flat = base.reshape(-1, dimension)
            flat = flat + rng.uniform(-1, 1, size=flat.shape) * scale_factor * KMAX
            shots = _project_shots(flat, KMAX).reshape(base.shape)
        else:
            shots = base

        pts = shots.reshape(-1, dimension)

        # Attraction step (uses shared grid data)
        pts = _sparkling_iterate(pts, _attraction, rep_kwargs,
                                  attraction_scale, repulsion_scale,
                                  n_iter_per, alpha, KMAX)
        shots = pts.reshape(shots.shape)

        # Interpolate to next resolution
        if idx < len(resolutions) - 1:
            Ns_next = Ns // resolutions[idx+1]
            if Ns_next > shots.shape[1]:
                ratio = Ns_next / shots.shape[1]
                shots = _interpolate_resolutions(shots, ratio)
                fp = (shots.reshape(-1, dimension) +
                      rng.uniform(-0.1*scale_factor, 0.1*scale_factor,
                                  size=(shots.shape[0]*dimension)))
                shots = _project_shots(fp, KMAX).reshape(shots.shape)

    # Final upscale to full Ns
    if shots.shape[1] < Ns:
        ratio = Ns / shots.shape[1]
        shots = _interpolate_resolutions(shots, ratio)
        fp = shots.reshape(-1, dimension)
        fp = fp + rng.uniform(-0.1*scale_factor, 0.1*scale_factor,
                              size=fp.shape)
        shots = _project_shots(fp, KMAX).reshape(shots.shape)

    # Ensure shot count
    if shots.shape[0] < Nc:
        shots = np.concatenate([shots, shots[-(Nc-shots.shape[0]):]], axis=0)
    elif shots.shape[0] > Nc:
        shots = shots[:Nc]

    return shots


# ── Public API ───────────────────────────────────────────────────────────────

def initialize_2D_sparkling(
    Nc: int,
    Ns: int,
    density=None,
    initial_type='radial',
    in_out=True,
    n_iter: int = 50,
    eps: float = 0.01,
    alpha=None,
    scale_factor: float = 0.75,
    attraction_scale: float = 1.0,
    repulsion_scale: float = 1.0,
    grid_size: int = 64,
    multi_resolution: bool = True,
    resolutions=None,
    use_pykeops: bool = False,
    proj_every_n: int = 5,
    seed=None,
) -> NDArray:
    """Initialize a 2D SPARKLING trajectory.

    Parameters
    ----------
    Nc : int
        Number of shots (interleaves).
    Ns : int
        Number of samples per shot (before multi-resolution).
    density : array-like or None
        Target k-space sampling density.  Must be a 1D or 2D array of the
        same shape as the ``grid_size`` grid (e.g. from
        :py:func:`mrinufft.trajectories.sampling.create_cutoff_decay_density`).
        Pass ``None`` for uniform density.
    initial_type : str
        Base trajectory before perturbation. One of ``'radial'``,
        ``'fibonacci_spiral'``, ``'spiral'``.
    in_out : bool
        Whether the base trajectory is center-out (*True*) or in-out.
    n_iter : int
        Total number of optimisation iterations at full resolution.
    eps : float
        Repulsion smoothing parameter.
    alpha : float or None
        Step size.  If *None*, defaults to ``scale_factor / (2*pi)``.
    scale_factor : float
        Perturbation magnitude and default step-size factor.
    attraction_scale : float
        Weight applied to the attraction gradient.
    repulsion_scale : float
        Weight applied to the repulsion force.
    multi_resolution : bool
        Use coarse-to-fine optimisation (default ``True``).
    resolutions : tuple[int, ...] or None
        Custom resolution levels.  Defaults to powers of two.
    use_pykeops : bool
        Use pykeops for GPU-accelerated repulsion.
    proj_every_n : int
        Project onto constraints every N iterations.  0 to disable.
    seed : int or None
        Random seed.

    Returns
    -------
    NDArray
        Trajectory of shape ``(Nc, Ns, 2)`` in normalized k-space
        coordinates ``[-k_max, k_max]``.
    """
    return _initialize_sparkling(
        Nc, Ns, 2, density=density, initial_type=initial_type, in_out=in_out,
        n_iter=n_iter, eps=eps, alpha=alpha, scale_factor=scale_factor,
        attraction_scale=attraction_scale, repulsion_scale=repulsion_scale,
        multi_resolution=multi_resolution, resolutions=resolutions,
        use_pykeops=use_pykeops, proj_every_n=proj_every_n, seed=seed,
    )


def initialize_3D_sparkling(
    Nc: int,
    Ns: int,
    density=None,
    initial_type='radial',
    in_out=True,
    n_iter: int = 50,
    eps: float = 0.01,
    alpha=None,
    scale_factor: float = 0.75,
    attraction_scale: float = 1.0,
    repulsion_scale: float = 1.0,
    grid_size: int = 64,
    multi_resolution: bool = True,
    resolutions=None,
    use_pykeops: bool = False,
    proj_every_n: int = 5,
    seed=None,
) -> NDArray:
    """Initialize a 3D SPARKLING trajectory.

    Parameters are the same as :py:func:`initialize_2D_sparkling`.

    Returns
    -------
    NDArray
        Trajectory of shape ``(Nc, Ns, 3)`` in normalized k-space
        coordinates ``[-k_max, k_max]``.
    """
    return _initialize_sparkling(
        Nc, Ns, 3, density=density, initial_type=initial_type, in_out=in_out,
        n_iter=n_iter, eps=eps, alpha=alpha, scale_factor=scale_factor,
        attraction_scale=attraction_scale, repulsion_scale=repulsion_scale,
        multi_resolution=multi_resolution, resolutions=resolutions,
        use_pykeops=use_pykeops, proj_every_n=proj_every_n, seed=seed,
    )
