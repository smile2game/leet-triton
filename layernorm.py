"""
layernorm 计算:
在 样本内部 做 归一化,在 最后一维度的 hidden dim上 做归一化。
也就是 每句话的每个token，在自己的 hidden states上做 归一化 
y = (x-u)/sqrt(v + eps) * w + b
"""

import triton
import triton.language as tl
import torch


@triton.jit
def _layer_norm_fwd(
    x_ptr,        # (M, N)
    w_ptr,        # (N,)
    b_ptr,        # (N,)
    y_ptr,        # (M, N)
    M, N,         # 行数/列数
    eps,          # float
    BLOCK_SIZE: tl.constexpr,  # = N
):
    row = tl.program_id(0)                     # 一行一个 program
    offsets = tl.arange(0, BLOCK_SIZE)            # 这一行内的列索引 [0..N)
    row_start = row * N
    mask = offsets < N                            # 简单安全：万一 N 不是 BLOCK_SIZE

    x = tl.load(x_ptr + row_start + offsets, mask=mask)

    #diff = x-u
    #rstd = 1/sqrt(var)
    mean = tl.sum(x, axis=0) / BLOCK_SIZE
    diff = x - mean
    var  = tl.sum(diff * diff, axis=0) / BLOCK_SIZE
    rstd = 1.0 / tl.sqrt(var + eps)
    # y = diff * rstd * w + b
    w = tl.load(w_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = diff * rstd * w + b

    tl.store(y_ptr + row_start + offsets, y, mask=mask)


def layer_norm(x, weight, bias, eps):
    # 最简单版本：一行一个 program，BLOCK=隐藏维
    M, N = x.shape #(1151,8192)
    y = torch.empty_like(x)
    BLOCK_SIZE = N
    grid = (M,)  # 启动 M 个 program

    _layer_norm_fwd[grid](
        x, weight, bias, y,
        M, N, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4, num_stages=2,
    )
    return y


if __name__ == "__main__":

    M = 1151
    N = 8192 
    dtype = torch.float32
    eps = 1e-5
    device = 'cuda'

    x_shape = (M,N) #(b*s, dim) 
    w_shape = (x_shape[-1],)

    weight = torch.rand(w_shape,dtype = dtype,device=device,requires_grad=True)
    bias = torch.rand(w_shape,dtype = dtype,device=device,requires_grad=True)
    #均值为 -2.3，方差为0.5，shape 为 (1151,8192)
    x = -2.3 + 0.5 * torch.randn(x_shape,dtype=dtype,device=device)
    y = layer_norm(x,weight,bias,eps)
    print(f"y is {y},\n shape is {x.shape}")

    import torch.nn.functional as F
    y_torch = F.layer_norm(x, normalized_shape=(N,), weight=weight, bias=bias, eps=eps)
    print(f"y_torch is {y_torch},\n shape is {y_torch.shape}")
    assert torch.allclose(y, y_torch, rtol=1e-3, atol=1e-3), \
    f"max abs diff = {(y - y_torch).abs().max().item()}"