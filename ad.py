import triton
import triton.language as tl
import torch

@triton.jit
def add_kernel(x_ptr,y_ptr,output_ptr,n_elements, BLOCK_SIZE: tl.constexpr): #block作为元
    #pid
    pid = tl.program_id(axis = 0) #获取 blockidx.x
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0,BLOCK_SIZE) #thread 作为元，这里是一个offsets集合，一写就是一行 ，也就是 一个 program
    mask = offsets < n_elements #每个 thread 都有自己的mask bool，这也是 bool列表。处理非整数倍 

    x = tl.load(x_ptr + offsets,mask=mask) #加载数据，从全局内存 加载到 共享内存，可能也会到寄存器？
    y = tl.load(y_ptr + offsets,mask=mask)
    output = x+y

    tl.store(output_ptr + offsets,output, mask = mask)

def add(x,y):
    output = torch.ones_like(x)
    n_elements = output.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )  #grid返回是一个元组注意,  meta是输入，n_elements是参数传入 
    add_kernel[grid](x,y,output,n_elements,BLOCK_SIZE=1024,num_warps = 4,num_stages = 2)  #grid规定操作，参数全进去，然后meta后三个
    return output

if __name__ == "__main__":
    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size,device='cuda')
    y = torch.rand(size,device='cuda')
    output_torch = x+y
    print(output_torch)

    output_triton = add(x,y)
    print(output_triton)

    assert torch.equal(output_triton, output_torch)

