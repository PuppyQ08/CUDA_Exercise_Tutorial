# How to compile
It requires g++, CUDA, CuRand, and nvprof (if profiler being used) installed. 
- Typing `make picuda` will compile the `pi.cu` file and type `./picuda` to run Pi MC evaluation in CUDA.
- Typing `make picpu` will compile the `pi_cpu.cpp` file and type `./picpu` to run Pi MC evaluation in CPU.

# Strategy
Inialiy I generate random number in host CPU and copy the two vector of random numbers to GPU device. Then use sequencial addressing reduction tree to collect all the count of dotsfrom each thread and atomic add to the output integer. But the CUDA profiler showed the memory copy process took 98% time of the whole calculation, which is about 3 secs with large overhead.
So I eliminate the memory copy process by using CuRand in each thread and do reduction to collect the summation. It turns out to be really effective and save space as well.

Possible optimization:
Sequencial addressing reduction tree is not the best way. Nowadays we have warp shuffling or unrolled the last warp reduction approach working better. 
Also atomic add in the final step to collect the sum from each block is not optimal. We can launch an another kernel to do one more reduction to eliminate the serial atomic add process.

# Pi MC in CPU result:
|      | Time/s  | Error     | 
|:---: | :---:   | :---:     |
|Run 1 | 2.81906 | 0.00012954|
|Run 2 | 2.82470 | 0.00012954|
|Run 3 | 2.83191 | 0.00012954|

# Pi MC in GPU result:
|      | Time/s  | Error     | 
|:---: | :---:   | :---:     |
|Run 1 | 0.32748 | 0.00004220|
|Run 2 | 0.33303 | 0.00004214|
|Run 3 | 0.29000 | 0.00004221|
