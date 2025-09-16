# leet-triton



满血 4060笔记本芯片 性能：

fp32 TFLOPS = 15 TFLOPS





Q：

1. triton调用时候的grid如何理解？

```python
grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )  #grid返回是一个元组注意,  meta是输入，n_elements是参数传入 
add_kernel[grid](x,y,output,n_elements,BLOCK_SIZE=1024,num_warps = 4,num_stages = 2)  #grid规定操作，参数全进去，然后meta后三个
```



2. triton的 级别映射

   1. cuda ： grid >> block >> thread
   2. triton:   grid >> program >> thread

      1. program : tl.program_id(axis)
      2. (thread) : tl.arrange(0,BLOCK_SIZE)   >> 隐式的

