"""Test the trajectories io module."""

import numpy as np
from mrinufft.io import read_trajectory, write_trajectory
from mrinufft.io.utils import add_phase_to_kspace_with_shifts
from mrinufft.trajectories.trajectory2D import initialize_2D_radial
from mrinufft.trajectories.trajectory3D import initialize_3D_cones
from pytest_cases import parametrize, parametrize_with_cases
from case_trajectories import CasesTrajectories


class CasesIO:
    """Cases 2 for IO tests, each has different parameters."""

    def case_trajectory_2D(self):
        """Test the 2D trajectory."""
        trajectory = initialize_2D_radial(
            Nc=32, Ns=256, tilt="uniform", in_out=False
        ).astype(np.float32)
        return "2D", trajectory, (0.23, 0.23), (256, 256), False, 2, 42.576e3, 1.1

    def case_trajectory_3D(self):
        """Test the 3D Trajectory."""
        trajectory = initialize_3D_cones(
            Nc=32, Ns=256, tilt="uniform", in_out=True
        ).astype(np.float32)
        return (
            "3D",
            trajectory,
            (0.23, 0.23, 0.1248),
            (256, 256, 128),
            True,
            5,
            10e3,
            1.2,
        )


@parametrize_with_cases(
    "name, trajectory, FOV, img_size, in_out, min_osf, gamma, recon_tag",
    cases=CasesIO,
)
@parametrize("version", [4.2, 5.0])
def test_write_n_read(
    name,
    trajectory,
    FOV,
    img_size,
    in_out,
    min_osf,
    gamma,
    recon_tag,
    tmp_path,
    version,
):
    """Test function which writes the trajectory and reads it back."""
    write_trajectory(
        trajectory=trajectory,
        FOV=FOV,
        img_size=img_size,
        check_constraints=True,
        grad_filename=str(tmp_path / name),
        in_out=in_out,
        version=version,
        min_osf=min_osf,
        recon_tag=recon_tag,
        gamma=gamma,
    )
    read_traj, params = read_trajectory(
        str((tmp_path / name).with_suffix(".bin")), gamma=gamma, read_shots=True
    )
    assert params["version"] == version
    assert params["num_shots"] == trajectory.shape[0]
    assert params["num_samples_per_shot"] == trajectory.shape[1] - 1
    assert params["TE"] == (0.5 if in_out else 0)
    assert params["gamma"] == gamma
    assert params["recon_tag"] == recon_tag
    assert params["min_osf"] == min_osf
    np.testing.assert_almost_equal(params["FOV"], FOV, decimal=6)
    np.testing.assert_equal(params["img_size"], img_size)
    np.testing.assert_almost_equal(read_traj, trajectory, decimal=5)


@parametrize_with_cases(
    "kspace_loc, shape",
    cases=[CasesTrajectories.case_random2D, CasesTrajectories.case_random3D],
)
def test_add_shift(kspace_loc, shape):
    """Test the add_phase_to_kspace_with_shifts function."""
    n_samples = np.prod(kspace_loc.shape[:-1])
    kspace_data = np.random.randn(n_samples) + 1j * np.random.randn(n_samples)
    shifts = np.random.rand(kspace_loc.shape[-1])

    shifted_data = add_phase_to_kspace_with_shifts(kspace_data, kspace_loc, shifts)

    assert np.allclose(np.abs(shifted_data), np.abs(kspace_data))

    phase = np.exp(-2 * np.pi * 1j * np.sum(kspace_loc * shifts, axis=-1))
    np.testing.assert_almost_equal(shifted_data / phase, kspace_data, decimal=5)
