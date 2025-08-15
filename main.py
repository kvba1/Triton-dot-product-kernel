import torch
import triton
import triton.testing
from dot_kernel import dot

SIZE = 1024
torch.manual_seed(0)
DEVICE = torch.device('cuda')

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # Argument names to use as an x-axis for the plot.
        x_vals=[2**i for i in range(0, 24)],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['triton', 'torch'],  # Possible values for `line_arg`.
        line_names=['Triton', 'Torch'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-')],  # Line styles.
        ylabel='GB/s',  # Label name for the y-axis.
        plot_name='dot-product-performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))

def benchmark(size, provider):
    x = torch.rand(size, device=DEVICE, dtype=torch.float32)
    y = torch.rand(size, device=DEVICE, dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.dot(x, y), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: dot(x, y), quantiles=quantiles)
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)

x = torch.randn(SIZE, dtype=torch.float32, device=DEVICE)
y = torch.randn(SIZE, dtype=torch.float32, device=DEVICE)

result_triton = dot(x, y)
result_torch = torch.dot(x, y)
assert torch.allclose(result_triton, result_torch), "Results do not match!"
print("Torch dot product result:", result_torch.item())
print("Dot product result:", result_triton.item())
print("Running benchmarks...")
benchmark.run(print_data=True, show_plots=True)