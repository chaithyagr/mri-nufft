"""Generate and visualize a 2D SPARKLING k-space trajectory.

The SPARKLING trajectory is produced by the algorithm described in
[`Cha+22`]_ via iterative gradient descent on a flat ``[Nc*Ns, D]`` array:
    1. Start from a base trajectory (radial / spiral) with perturbed samples
    2. Iteratively update using attraction (toward target density) and
       repulsion (prevent clustering) forces
    3. Periodically project onto hardware constraints (gradient / slew bounds)
    4. Use coarse-to-fine multi-resolution scheme

Parameters
-----
* ``density`` – target k-space density array (e.g. from
  ``mrinufft.trajectories.sampling.create_cutoff_decay_density``)
* ``initial_type`` – base trajectory before perturbation
  (``'radial'``, ``'fibonacci_spiral'``, ``'spiral'``)
* ``multi_resolution`` – coarse-to-fine optimisation (``True`` by default)
* ``proj_every_n`` – project onto constraints every N iterations

Example usage
-----
.. code-block:: python

    from mrinufft.trajectories.inits.sparkling import initialize_2D_sparkling
    from mrinufft.trajectories.sampling import create_cutoff_decay_density

    # 1. Create target density grid
    density = create_cutoff_decay_density(
        shape=(64, 64), cutoff=25.0, decay=2.0
    )

    # 2. Generate SPARKLING trajectory
    traj = initialize_2D_sparkling(
        Nc=64, Ns=256, density=density.reshape(-1),
        initial_type='radial', multi_resolution=True, seed=42
    )
    # traj shape: (64, 256, 2), values in [-0.5, 0.5]

References
-----
.. [`Cha+22`] Chaithya, G. R., Pierre Weiss, Guillaume Daval-Frérot,
    Aurélien Massire, Alexandre Vignaud, and Philippe Ciuciu.
    "Optimizing full 3D SPARKLING trajectories for high-resolution
    magnetic resonance imaging." IEEE Trans. Med. Imag. 41(8), 2105–2117 (2022).
"""

import numpy as np

from mrinufft.trajectories.inits.sparkling import initialize_2D_sparkling
from mrinufft.trajectories.sampling import create_cutoff_decay_density


# ── constants ──────────────────────────────

Nc = 64          # number of shots (interleaves)
Ns = 256         # number of samples per shot (before multi-res)
grid_size = 64   # resolution of the target-density grid
seed = 42        # reproducibility


def main():
    # ── 1. Build target density from mrinufft's sampling module ──
    density = create_cutoff_decay_density(
        shape=(grid_size, grid_size),
        cutoff=25.0,   # central plateau ratio
        decay=2.0,     # polynomial decay exponent
    )

    # ── 2. Generate SPARKLING trajectory ──
    traj = initialize_2D_sparkling(
        Nc=Nc, Ns=Ns,
        density=density.reshape(-1),   # density must be a 1-D array
        initial_type='radial',
        in_out=True,
        n_iter=100,
        eps=0.01,
        scale_factor=0.75,
        multi_resolution=True,
        seed=seed,
    )

    print(f"Trajectory shape : {traj.shape}")
    print(f"Max norm         : {np.linalg.norm(traj, axis=-1).max():.4f}")
    print(f"Min norm         : {np.linalg.norm(traj, axis=-1).min():.4f}")
    print(f"Mean norm        : {np.linalg.norm(traj, axis=-1).mean():.4f}")

    # ── 3. Visualise target density and trajectory ──
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Target k-space density (reshaped to 2-D)
    axes[0].imshow(density, origin='lower', extent=[-0.5, 0.5, -0.5, 0.5],
                   cmap='viridis', aspect='equal')
    axes[0].set_title("Target density\n(cutoff=25, decay=2)")
    axes[0].set_xlabel("kx")
    axes[0].set_ylabel("ky")

    # Optimized SPARKLING trajectory
    x, y = traj[:, :, 0].ravel(), traj[:, :, 1].ravel()
    axes[1].scatter(x, y, s=0.5, color='C1', alpha=0.3)
    axes[1].axis('equal')
    axes[1].set_xlim(-0.5, 0.5)
    axes[1].set_ylim(-0.5, 0.5)
    axes[1].set_title("SPARKLING trajectory\n(64 shots × 256 samples)")
    axes[1].set_xlabel("kx")
    axes[1].set_ylabel("ky")

    fig.tight_layout()
    plt.savefig('sparkling_2d_example.png', dpi=150)
    print("\nSaved figure to sparkling_2d_example.png")
    plt.show()


if __name__ == '__main__':
    main()
