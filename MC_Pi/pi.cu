/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <iostream>
#include <vector>
#include <ctime>
#include <math.h>
#include <curand_kernel.h>
#include <curand.h>
#define PI 3.1415926535897931f
#define BLOCKSIZE 128 
//128 threads/blocks 625 random number per threads, 375 blocks.

static int samples = 30e6;

__global__ void picuda(curandState *state,int* output,int numRand,int len,int numblocks){
    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int idx = bx*blockDim.x+tx;
    //initialize the curand generator
    //I use same seed with different sequence for each thread
    curand_init(1234,idx,0,&state[idx]);
    //shared vector for storing sum from each thread
    __shared__ int sharedCount[BLOCKSIZE];
    // copy state to local memory for efficiency
    curandState localstate = state[idx];
    int countvalue = 0;
    //each thread generate "numRand" numbers randomly 
    for(int i = 0; i < numRand; i++){
        float x = curand_uniform(&localstate);
        float y = curand_uniform(&localstate);
        if(x*x + y*y <= 1.0f){
            ++countvalue;
        }
    }
    state[idx] = localstate;
    sharedCount[tx] = countvalue;
    __syncthreads();
    //Do reduction on each block
    for (unsigned int redstride = blockDim.x/2; redstride >= 1; redstride/=2){
            __syncthreads();
            if(tx < redstride)
                sharedCount[tx] += sharedCount[tx+redstride];
    }
    //atomic add each block's sum to the output integer 
    if(tx == 0)
        atomicAdd(output, sharedCount[0]);
}

int main(void){
    /*
      TODO any initialization code you need goes here, e.g. random
      number seeding, cudaMalloc allocations, etc.  Random number
      _generation_ should still go in pi().
    */
    std::clock_t start;
    double duration;
    start = std::clock();
    int* d_outint;
    int* h_outint;
    curandState *States;
    h_outint = new int;
    //thread number of each block
    int block_size = BLOCKSIZE;
    //total number of blocks, each thread run 625 random numbers.
    int num_blocks = ceil(samples/(block_size*625));
    int numRand = 625;
    printf("number of blocks %d \n",num_blocks);
    printf("number of random number per thread %d \n",numRand);
    //allocate variable in device
    cudaMalloc((void**)&d_outint, sizeof(int));
    cudaMalloc((void**)&States,block_size*num_blocks*sizeof(curandState));
    /*
      TODO Put your code here.  You can use anything in the CUDA
      Toolkit, including libraries, Thrust, or your own device
      kernels, but do not use ArrayFire functions here.  If you have
      initialization code, see pi_init().
    */
    //launch kernels
    picuda<<<num_blocks, block_size>>>(States,d_outint,numRand,samples,num_blocks);
    if ( cudaSuccess != cudaGetLastError())
        printf("Error!\n");

    //collect data from the device
    cudaMemcpy(h_outint, d_outint,sizeof(int),cudaMemcpyDeviceToHost);
    float outPi = 4.0 * *h_outint/samples;
    float error = fabs(PI - outPi);   
    
    duration = (std::clock()-start)/(double) CLOCKS_PER_SEC;
    printf("GPU runing for evaluating Pi took %7.5f sec and error of %.8f", duration, error);

    /*
      TODO This function should contain the clean up. You should add
      memory deallocation etc here.
    */
    cudaFree(d_outint);
    free(h_outint);

}
