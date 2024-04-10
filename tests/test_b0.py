"""Test the density compensation methods."""

import numpy as np
import numpy.testing as npt
from pytest_cases import parametrize, parametrize_with_cases

from case_trajectories import CasesTrajectories
from helpers import assert_correlate
from mrinufft.operators.off_resonnance import MRIFourierCorrected 
from mrinufft.density.utils import normalize_weights
from mrinufft._utils import proper_trajectory


from helpers import (
    kspace_from_op,
    image_from_op,
    to_interface,
    from_interface,
    param_array_interface,
)


@param_array_interface
def test_interfaces_autoadjoint(operator, array_interface):
    """Test the adjoint property of the operator."""
    reldiff = np.zeros(10)
    for i in range(10):
        img_data = to_interface(image_from_op(operator), array_interface)
        ksp_data = to_interface(kspace_from_op(operator), array_interface)
        kspace = operator.op(img_data)

        rightadjoint = np.vdot(
            from_interface(kspace, array_interface),
            from_interface(ksp_data, array_interface),
        )

        image = operator.adj_op(ksp_data)
        leftadjoint = np.vdot(
            from_interface(img_data, array_interface),
            from_interface(image, array_interface),
        )
        reldiff[i] = abs(rightadjoint - leftadjoint) / abs(leftadjoint)
    print(reldiff)
    assert np.mean(reldiff) < 5e-5