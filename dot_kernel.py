import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def dot_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    #one kernel per 1024 vector elements
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = tl.sum(x * y)

    if pid == 0:
        tl.store(output_ptr, output)
    else:
        tl.atomic_add(output_ptr, output)

def dot(x :torch.Tensor, y :torch.Tensor) -> torch.Tensor:
    assert x.dim() == 1 and y.dim() == 1, 'Input tensors must be 1D'
    assert x.shape == y.shape, 'Tensors must be of the same shape'
    output = torch.empty((), dtype=x.dtype, device=x.device)
    n = x.numel()

    # meta
    block_size = 1024
    grid = lambda meta: (triton.cdiv(n, block_size),)

    dot_kernel[grid](x, y, output, n, block_size)

    return output
    
