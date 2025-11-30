#Lesson 1 [How to profile CUDA kernels in PyTorch](https://www.youtube.com/watch?v=LuhJEEJQgUM)

Triton kernel can be slower than Torch if the BLOCKSIZE is not proper number.

Using Torch compile TORCH_LOGS="output_code" can generate the triton code automatically.

#Lesson 2 [Computer and memory](https://www.youtube.com/watch?v=lTmYrKwjSOU)

After Volta architecture, implicit lockstep doesn't exist within warp so there is no automatic reconvergence. We need to use __syncwarp to explicitly sync threads in warp. (Can check from [Volta whitepaper](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf?utm_source=chatgpt.com)

