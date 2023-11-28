""""Test the gradients of the forward and adjoint operations"""
import pytest
import torch
import numpy as np
from mrinufft import get_operator


def get_fourier_matrix(ktraj, im_size, im_rank, do_ifft=False):
    r = [torch.linspace(-im_size[i]/2, im_size[i]/2-1, im_size[i]) for i in range(im_rank)]
    grid_r = torch.reshape(torch.stack(torch.meshgrid(*r ,indexing='ij')), (im_rank, torch.prod(torch.tensor(im_size))))
    traj_grid = torch.matmul(ktraj, grid_r)
    if do_ifft:
        A = torch.exp(1j * traj_grid)
    else:
        A = torch.exp(-1j * traj_grid)
    A = A / (np.sqrt(torch.prod(torch.tensor(im_size))) * np.power(np.sqrt(2), im_rank))
    return A


@pytest.mark.parametrize('im_size', [(10, 10)])
def test_adjoint_and_gradients(im_size):
    torch.random.manual_seed(0)
    im_rank = len(im_size)
    M = im_size[0] * 2**im_rank
    # Generate Trajectory
    kspace_loc = (torch.rand(M, im_rank) - 0.5)*2*np.pi
    nufft_op = get_operator('cufinufft')(
        kspace_loc, 
        im_size, 
        1,
    )
    nuifft_matrix = get_fourier_matrix(kspace_loc, im_size, im_rank, do_ifft=True).cuda()
    nufft_matrix = get_fourier_matrix(kspace_loc, im_size, im_rank, do_ifft=False).cuda()
    
    image = torch.rand(*im_size).cuda().type(torch.complex64)
    
    kspace_data = nufft_op.op(image)
    kspace_data_ndft = torch.matmul(image.flatten(), nufft_matrix.t())
    
    adj_image = nufft_op.adj_op(kspace_data)
    adj_image_ndft = torch.reshape(torch.transpose(torch.matmul(kspace_data, nuifft_matrix), [0, 1, 2]), (batch_size, 1, *im_size))
    
    
    # Test gradients with respect to kdata
    gradient_ndft_kdata = g.gradient(I_ndft, kdata)[0]
    gradient_nufft_kdata = g.gradient(I_nufft, kdata)[0]
    torch.test.assertAllClose(gradient_ndft_kdata, gradient_nufft_kdata, atol=6e-3)

    # Test gradients with respect to trajectory location
    gradient_ndft_traj = g.gradient(I_ndft, ktraj)[0]
    gradient_nufft_traj = g.gradient(I_nufft, ktraj)[0]
    torch.test.assertAllClose(gradient_ndft_traj, gradient_nufft_traj, atol=6e-3)

    # Test gradients in chain rule with respect to ktraj
    gradient_ndft_loss = g.gradient(loss_ndft, ktraj)[0]
    gradient_nufft_loss = g.gradient(loss_nufft, ktraj)[0]
    torch.test.assertAllClose(gradient_ndft_loss, gradient_nufft_loss, atol=5e-4)

    # This is gradient of NDFT from matrix, will help in debug
    # gradient_from_matrix = 2*np.pi*1j*torch.matmul(torch.cast(r, torch.complex64), torch.transpose(A))*kdata[0][0]

