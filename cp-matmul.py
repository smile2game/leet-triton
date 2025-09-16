import triton
import triton.language as tl
import torch

from torch.profiler import profile,record_function,ProfilerActivity


@triton.jit
def _matmul_kernel(
    a_ptr, b_ptr, c_ptr,            # A: (M,K), B: (K,N), C: (M,N)
    M, N, K,                         # 维度
    stride_am, stride_ak,            # A 的步长: (am, ak)
    stride_bk, stride_bn,            # B 的步长: (bk, bn)
    stride_cm, stride_cn,            # C 的步长: (cm, cn)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 2D 网格：pid_m 控制行块，pid_n 控制列块
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # 当前块内的行、列偏移
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M) #
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # 累加器：用 fp32 做累加，易懂且稳定
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # A/B 的起始指针（指向当前 K 块）
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am) + (offs_k[None, :] * stride_ak) #[BM,1] + [1,BK]
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk) + (offs_n[None, :] * stride_bn)

    # 沿着 K 维分块累加
    for k0 in range(0, K, BLOCK_K):
        # 有效性 mask（边界超出就置 0）
        a_mask = (offs_m[:, None] < M) & ((k0 + offs_k)[None, :] < K)
        b_mask = ((k0 + offs_k)[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # 最直接的块乘（矩阵乘）累加
        acc += tl.dot(a, b,allow_tf32=False)

        # 指针前进到下一段 K
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # 写回 C：把 fp32 累加结果 cast 回输入 dtype（这里按 A 的 dtype 回写）
    c = acc.to(tl.float32)  # 你也可以 cast 成别的 dtype
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(
        c_ptr + (offs_m[:, None] * stride_cm) + (offs_n[None, :] * stride_cn),
        c,
        mask=c_mask
    )


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    最简单版本：tile + K 维循环累加。
    A: (M, K), B: (K, N), 返回 C: (M, N)
    """

    M, K = a.shape
    K2, N = b.shape
    # 输出
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    print(f"a.stride(0) is {a.stride(0)}, a.stride(1) is {a.stride(1)}")
    # 简单固定的 tile 大小；不花哨
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N)) #grid = (2048//64 ,1024//64) = (32，16) >> grid是数据尺寸，也就是 grid size 表示有多少个 block
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

if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda"

    # 关键：关闭 TF32（否则 torch 的 matmul 会比 fp32 更粗）
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    M, K, N = 2048, 1024, 512   # 随便取个不是 64 的倍数，看看边界 mask 是否正确
    a = torch.randn(M, K, device=device, dtype=torch.float32) 
    b = torch.randn(K, N, device=device, dtype=torch.float32) 

    with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],
                 record_shapes=True,
                 with_stack=True) as prof:
        with record_function("matmul_host_call"):
            
            c_triton = matmul(a, b)
            c_torch = a @ b

            torch.cuda.synchronize()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    prof.export_chrome_trace("trace/trace_matmul.json")

    print(f"c_triton is {c_triton},\n shape is {c_triton.shape}")
    print(f"c_torch is {c_torch},\n shape is {c_torch.shape}")

    print("max abs diff:", (c_triton - c_torch).abs().max().item())
    assert torch.allclose(c_triton, c_torch, rtol=1e-3, atol=1e-3)
    print("OK!")
