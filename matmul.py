import os
import triton, triton.language as tl
import torch
from statistics import mean, median
from torch.profiler import profile, record_function, ProfilerActivity
import torch.cuda.nvtx as nvtx

# ---- 你的 kernel & 包装函数（原样，不改计算路径） ----
@triton.jit
def _matmul_kernel(a_ptr, b_ptr, c_ptr,
                   M, N, K,
                   stride_am, stride_ak,
                   stride_bk, stride_bn,
                   stride_cm, stride_cn,
                   BLOCK_M: tl.constexpr,
                   BLOCK_N: tl.constexpr,
                   BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am) + (offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk) + (offs_n[None, :] * stride_bn)

    for k0 in range(0, K, BLOCK_K):
        a_mask = (offs_m[:, None] < M) & ((k0 + offs_k)[None, :] < K)
        b_mask = ((k0 + offs_k)[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b, allow_tf32=False)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(tl.float32)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptr + (offs_m[:, None] * stride_cm) + (offs_n[None, :] * stride_cn), c, mask=c_mask)

def triton_matmul(a, b):
    M, K = a.shape
    K2, N = b.shape
    assert K2 == K
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return c

# ---- 计时工具：用 CUDA Events 测 “GPU 纯执行时间” ----
def gpu_time_ms(fn, *args, iters=20, warmup=10):
    # 预热（不计时）：触发 Triton JIT / cudnn autotune / 内存池等
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        fn(*args)              # 仅测这一行在 GPU 上的执行时长
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))  # ms
    return times

# ---- 用 PyTorch Profiler 捕捉 kernel 执行时间并导出 trace ----
def profile_once_matmul(label, fn, a, b, trace_path):
    os.makedirs(os.path.dirname(trace_path), exist_ok=True)
    # 预热一遍，避免把 JIT/内存分配算进采样窗口
    _ = fn(a, b)
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True, with_stack=True) as prof:
        with record_function(f"{label}_matmul_step"):
            nvtx.range_push(f"{label}_matmul")
            _ = fn(a, b)
            nvtx.range_pop()
        # ☆ 关键：在 profile 作用域内同步，确保记录到完整的 GPU kernel 时间
        torch.cuda.synchronize()

    print(f"\n[{label}] top CUDA kernels by time:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    prof.export_chrome_trace(trace_path)
    print(f"[{label}] Chrome/Perfetto trace saved: {trace_path}")

if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # 你脚本里的尺寸
    M, K, N = 2048, 1024, 512
    a = torch.randn(M, K, device=device, dtype=torch.float32)
    b = torch.randn(K, N, device=device, dtype=torch.float32)

    # 先各自跑一次，确保 Triton 已编译（避免首次调用被计入）
    _ = triton_matmul(a, b)
    _ = a @ b
    torch.cuda.synchronize()

    # ------------ CUDA Events：纯核实测 ------------
    t_triton = gpu_time_ms(triton_matmul, a, b, iters=10, warmup=5)
    t_torch  = gpu_time_ms(lambda x, y: x @ y, a, b, iters=10, warmup=5)

    # FLOPs（2*M*K*N）
    flops = 2.0 * M * K * N
    def summarize(name, times_ms):
        avg = mean(times_ms); med = median(times_ms)
        tflops_avg = flops / (avg/1e3) / 1e12
        tflops_med = flops / (med/1e3) / 1e12
        print(f"{name}: iters={len(times_ms)}")
        print(f"  avg  {avg:8.3f} ms   |  median {med:8.3f} ms")
        print(f"  TFLOPs (avg)  : {tflops_avg:6.2f}")
        print(f"  TFLOPs (med)  : {tflops_med:6.2f}")

    summarize("Triton", t_triton)
    summarize("Torch @", t_torch)

    spd_avg = (mean(t_torch) / mean(t_triton))
    spd_med = (median(t_torch) / median(t_triton))
    print(f"\nSpeedup (Torch / Triton):  avg×{spd_avg:.2f}   median×{spd_med:.2f}")

    # 正确性快速校验
    out_t = triton_matmul(a, b)
    out_p = a @ b
    print("max abs diff:", (out_t - out_p).abs().max().item())

    # ------------ PyTorch Profiler：导出 trace ------------
    profile_once_matmul("TRITON", triton_matmul, a, b, trace_path="trace/triton_matmul.json")
    profile_once_matmul("TORCH",  lambda x, y: x @ y, a, b, trace_path="trace/torch_matmul.json")