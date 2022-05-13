# CUDA_Exercise_Tutorial
------------------------
A practical set of exercise for CUDA GPU parallel programming

## Dependence
- Linux system with valid Nvidia GPU and drive
- gcc, cmake, cuda-11.6

## How to Make
- CPU running file
```
make cpu
``` 
- GPU running file
```
make cu
```
- Makefile explain\
`--ccbin`: to set the host compiler as g++ (default)\
`-I`: include the path of any cuda related library\
`-m64`: specific 64 bit (default)\
`-g`: to generate and embed debug information\
`--thread 0`:Specify the maximum number of threads to be used to execute the compilation steps in parallel. This option can be used to improve the compilation speed when compiling for multiple architectures. The compiler creates number threads to execute the compilation steps in parallel. If number is 1, this option is ignored. If number is 0, the number of threads used is the number of CPUs on the machine. (default)\
`-gencode arch=compute_XX, code=sm_XX`:  XX is the compute capability for the GPU you are using, like for GTX 960M based on Maxwell arch is SM5.0 so XX= 50\
When two XXs are different, here is what happened:
	- A temporary PTX code will be generated from your source code, and it will use ccXX1 PTX.
	- From that PTX, the `ptxas` tool will generate ccXX2-compliant SASS code.
	- The SASS code will be embedded in your executable.
	- The PTX code will be discarded.
[See what is PTX and ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)

## Know your device
[The NVIDIA CUDA samples repository](https://github.com/NVIDIA/cuda-samples) offers a really good instruction on how to catch on coding on GPU by CUDA.
In the 1_Utility directory, we can find the device query code to inspect your GPU device. (A more detailed inspection can be obtained from Nsight Compute).
Here is the output:
```
Detected 1 CUDA Capable device(s)
Device 0: "NVIDIA GeForce GTX 960M"
  CUDA Driver Version / Runtime Version          11.6 / 11.6
  CUDA Capability Major/Minor version number:    5.0
  Total amount of global memory:                 2004 MBytes (2101870592 bytes)
  (005) Multiprocessors, (128) CUDA Cores/MP:    640 CUDA Cores
```
CUDA core count represents the total number of single precision floating point or integer thread instructions that can be executed per cycle.
One thread can run on one core but also on multiple cores. The concept of Block and thread is more related to how the memory is partitioned.
```
  GPU Max Clock rate:                            1176 MHz (1.18 GHz)
``` 
CPU can have 3+GHz clock rate. So GPU is low-clock rate but high throughput device.
```
  Memory Clock rate:                             2505 Mhz
  Memory Bus Width:                              128-bit
  L2 Cache Size:                                 2097152 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(65536), 2D=(65536, 65536), 3D=(4096, 4096, 4096)
  Maximum Layered 1D Texture Size, (num) layers  1D=(16384), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(16384, 16384), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total shared memory per multiprocessor:        65536 bytes
  ```
  The hardware max of number of thread blocks is defined from compute capability (like cc3.2 is 16).
  But it won't reach that limit if you have a huge block (large shared memory) as you can see shared memroy per block is more or less around the shared memory per SM.
  ```
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  2048
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 1 copy engine(s)
  Run time limit on kernels:                     Yes
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Disabled
  Device supports Unified Addressing (UVA):      Yes
  Device supports Managed Memory:                Yes
  Device supports Compute Preemption:            No
  Supports Cooperative Kernel Launch:            No
  Supports MultiDevice Co-op Kernel Launch:      No
```
