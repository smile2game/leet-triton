# filename: industrial_triton_linear_demo.py
import os, math, time
import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton import autotune, Config
from torch.profiler import profile, record_function, ProfilerActivity

# =========================================================
# 1) Triton GEMM kernel（含 autotune；fp32 累加 & IO）
# =========================================================
@autotune(
    configs=[
        Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=2),
        Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=8, num_stages=2),
        Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64},num_warps=8, num_stages=3),
        # 你可以继续加更多配置覆盖不同形状/显卡
    ],
    key=['M', 'N', 'K'],   # 以 (M,N,K) 为 key 缓存最快配置
)
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
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am) + (offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk) + (offs_n[None, :] * stride_bn)

    # K 维分块累加
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

def triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """A:(M,K), B:(K,N) -> C:(M,N)  （CUDA, fp32 输出）"""
    assert a.is_cuda and b.is_cuda and a.ndim == 2 and b.ndim == 2
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"Shape mismatch: A({M},{K}) x B({K2},{N})"
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))  # grid 仅用于 launch，BLOCK_* 由 autotune 决定
    _matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c

# =========================================================
# 2) TritonLinear（Autograd.Function + nn.Module）
# =========================================================
class _TritonLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        # x: (..., in_features), weight: (out_features, in_features)
        assert x.is_cuda and weight.is_cuda
        in_features  = weight.shape[1]
        out_features = weight.shape[0]

        x_shape = x.shape
        x2d = x.reshape(-1, in_features).contiguous()       # (M,K)
        Wt  = weight.t().contiguous()                       # (K,N)

        y2d = triton_matmul(x2d.to(torch.float32), Wt.to(torch.float32))  # (M,N) fp32
        if bias is not None:
            y2d = y2d + bias.to(torch.float32)

        # 保存反向所需
        ctx.save_for_backward(x2d, weight, bias if bias is not None else torch.tensor([], device=x.device))
        ctx.has_bias = bias is not None
        ctx.in_features = in_features
        ctx.out_features = out_features
        ctx.x_shape = x_shape
        ctx.x_dtype = x.dtype
        ctx.w_dtype = weight.dtype
        ctx.b_dtype = bias.dtype if bias is not None else None

        return y2d.to(x.dtype).reshape(*x_shape[:-1], out_features)

    @staticmethod
    def backward(ctx, grad_out):
        x2d, weight, bias_sentinel = ctx.saved_tensors
        has_bias = ctx.has_bias
        in_features  = ctx.in_features
        out_features = ctx.out_features
        x_shape = ctx.x_shape
        x_dtype, w_dtype, b_dtype = ctx.x_dtype, ctx.w_dtype, ctx.b_dtype

        grad_out_2d = grad_out.reshape(-1, out_features).contiguous().to(torch.float32)

        # dX = dY @ W
        grad_x_2d = triton_matmul(grad_out_2d, weight.to(torch.float32).contiguous())
        grad_x = grad_x_2d.to(x_dtype).reshape(*x_shape)

        # dW = dY^T @ X
        grad_w = triton_matmul(grad_out_2d.t().contiguous(), x2d.to(torch.float32)).to(w_dtype)

        # db = sum(dY, dim=0)
        grad_b = grad_out_2d.sum(dim=0).to(b_dtype) if has_bias else None

        return grad_x, grad_w, grad_b

class TritonLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.empty(out_features)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1.0 / math.sqrt(in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return _TritonLinearFn.apply(x, self.weight, self.bias)

# =========================================================
# 3) 两层 MLP：Triton 版 & Torch 版
# =========================================================
class TritonMLP(nn.Module):
    def __init__(self, din, dhid, dout):
        super().__init__()
        self.fc1 = TritonLinear(din, dhid, bias=True)
        self.act = nn.GELU()
        self.fc2 = TritonLinear(dhid, dout, bias=True)
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class TorchMLP(nn.Module):
    def __init__(self, din, dhid, dout):
        super().__init__()
        self.fc1 = nn.Linear(din, dhid, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dhid, dout, bias=True)
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

# =========================================================
# 4) 训练 & Profiling 工具函数
# =========================================================
def copy_params(dst: nn.Module, src: nn.Module):
    """把 src 的参数拷贝到 dst（形状、名字匹配优先，按层次 heuristic）"""
    with torch.no_grad():
        src_tensors = [p.data for p in src.parameters()]
        i = 0
        for p in dst.parameters():
            p.copy_(src_tensors[i].to(p.dtype).to(p.device))
            i += 1

def run_training_with_profiler(model: nn.Module, optimizer: torch.optim.Optimizer,
                               x: torch.Tensor, y: torch.Tensor,
                               steps: int, trace_path: str, label: str):
    os.makedirs(os.path.dirname(trace_path), exist_ok=True)
    criterion = nn.MSELoss()

    # 预热（触发 Triton JIT）
    _ = model(x)
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True, with_stack=True) as prof:
        with record_function(f"{label}_train_10_steps"):
            for step in range(steps):
                optimizer.zero_grad(set_to_none=True)
                y_pred = model(x)
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()
            torch.cuda.synchronize()

    print(f"[{label}] final loss:", loss.item())
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    prof.export_chrome_trace(trace_path)
    print(f"[{label}] Chrome trace saved to: {trace_path}")

# =========================================================
# 5) 主流程：4 条样本，10 次迭代，Triton vs Torch
# =========================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    device = "cuda"
    B = 4               # 四条数据
    Din = 1024
    Dhid = 2048
    Dout = 513          # 特意不设为 64 的倍数，触发 mask 路径
    steps = 10

    # 模拟数据（固定 4 条）
    x = torch.randn(B, Din, device=device, dtype=torch.float32)
    # 回归任务：目标 y
    y = torch.randn(B, Dout, device=device, dtype=torch.float32)

    # 构建两个模型：确保初始权重一致以便更公平对比
    torch_model  = TorchMLP(Din, Dhid, Dout).to(device).to(torch.float32)
    triton_model = TritonMLP(Din, Dhid, Dout).to(device)

    # 让 Triton 版初始参数与 Torch 版完全一致
    copy_params(triton_model, torch_model)

    # 优化器（相同超参）
    opt_torch  = torch.optim.SGD(torch_model.parameters(),  lr=1e-2, momentum=0.9)
    opt_triton = torch.optim.SGD(triton_model.parameters(), lr=1e-2, momentum=0.9)

    # 先跑 Triton，再跑 Torch（或反过来都行）
    run_training_with_profiler(
        triton_model, opt_triton, x, y, steps,
        trace_path="trace/trace_triton.json", label="TRITON"
    )
    run_training_with_profiler(
        torch_model, opt_torch, x, y, steps,
        trace_path="trace/trace_torch.json", label="TORCH"
    )
