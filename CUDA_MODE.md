#Lesson 1 [How to profile CUDA kernels in PyTorch](https://www.youtube.com/watch?v=LuhJEEJQgUM)

Triton kernel can be slower than Torch if the BLOCKSIZE is not proper number.

Using Torch compile TORCH_LOGS="output_code" can generate the triton code automatically.
