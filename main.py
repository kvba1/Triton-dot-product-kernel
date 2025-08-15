import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def relu_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    