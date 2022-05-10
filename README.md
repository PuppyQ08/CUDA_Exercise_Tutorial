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
	- From that PTX, the ```ptxas``` tool will generate ccXX2-compliant SASS code.
	- The SASS code will be embedded in your executable.
	- The PTX code will be discarded.
[See what is PTX and ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)

