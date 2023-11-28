"""Utilities for interfaces."""
from .utils import (
    check_error,
    check_size,
    sizeof_fmt,
)

from .gpu_utils import (
    get_maxThreadBlock,
    CUPY_AVAILABLE,
    is_host_array,
    is_cuda_array,
    pin_memory,
    nvtx_mark,
    change_type,
    is_torch_tensor,
    get_ptr,
)

__all__ = [
    "check_error",
    "check_size",
    "sizeof_fmt",
    "get_maxThreadBlock",
    "CUPY_AVAILABLE",
    "is_host_array",
    "is_cuda_array",
    "change_type",
    "is_torch_tensor",
    "pin_memory",
    "nvtx_mark",
    "get_ptr",
]
