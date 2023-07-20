"""Test the trajectories io module."""
import numpy as np
from mrinufft.trajectories.io import read_trajectory, write_trajectory
from mrinufft.trajectories.trajectory2D import initialize_2D_radial
from mrinufft.trajectories.trajectory3D import initialize_3D_cones
from pytest_cases import parametrize_with_cases


class CasesTrajectory:
    def trajectory_2D(self):
        """Test the 2D trajectory."""
        trajectory = initialize_2D_radial(
            Nc=32,
            Ns=256,
            tilt="uniform",
            in_out=False
        ).astype(np.float32)
        return "2D", trajectory, (0.23, 0.23), \
            (256, 256), False, 2, 42.576e3, 1.1
    def trajectory_3D(self):
        "Test the 3D Trajectory"
        trajectory = initialize_3D_cones(
            Nc=32,
            Ns=256,
            tilt="uniform",
            in_out=True
        ).astype(np.float32)
        return "3D", trajectory, (0.23, 0.23, 0.1248), \
            (256, 256, 128), True, 5, 10e3, 1.2

@parametrize_with_cases(
    argnames="name, trajectory, FOV, img_size, in_out, min_osf, gamma, recon_tag", 
    cases=CasesTrajectory
)
def test_write_n_read(name, trajectory, FOV, img_size,
                      in_out, min_osf, gamma, recon_tag):
    write_trajectory(
        trajectory=trajectory,
        FOV=FOV,
        img_size=img_size,
        check_constraints=True,
        grad_filename=name,
        in_out=in_out,
        version=4.2,
        min_osf=min_osf,
        recon_tag=recon_tag,
        gamma=gamma,
    )
    read_trajectory, params = read_trajectory(
        "test.bin",
        read_shots=True
    )
    assert params["version"] == 4.2
    assert params["num_shots"] == trajectory.shape[0]
    assert params["num_samples_per_shot"] == trajectory.shape[1]-1
    assert params["TE"] == (0.5 if in_out else 0)
    assert params["gamma"] == gamma
    assert params["recon_tag"] == recon_tag
    assert params["min_osf"] == min_osf
    np.testing.assert_equal(params["FOV"], FOV)
    np.testing.assert_equal(params["img_size"], img_size)
    np.testing.assert_almost_equal(read_trajectory, trajectory, decimal=6)
