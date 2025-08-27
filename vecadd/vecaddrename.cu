#include <stdio.h>
#include <cuda_runtime.h>

//very basic version of vec add, each thread do one adding.
__global__ void vecadd(const float *A, const float *B, float *C, int numEle){
    int  idx = blockDim.x*blockIdx.x + threadIdx.x;

    if (idx < numEle ) {
        C[idx] = A[idx] + B[idx];
    }
}


//host main routine
int main(void){
    cudaError_t err = cudaSuccess;
    int numEle = 50000;
    size_t size = numEle * sizeof(float);
    //host vec
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    for (int i = 0; i< numEle; ++i){
        h_A[i] = rand()/(float) RAND_MAX;
        h_B[i] = rand()/(float) RAND_MAX;
    }
    //device vec
    float *d_A = NULL;
    err = cudaMalloc((void **) &d_A, size);

    if(err != cudaSuccess){
        fprintf(stderr, "fail to allocate device vec A (errorcode %s )!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);
    }

    float *d_B = NULL;
    err = cudaMalloc((void **) &d_B, size);
    float *d_C = NULL;
    err = cudaMalloc((void **) &d_C, size);
    //mem copy to device.
    cudaMemcpy(d_A,h_A,size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C,h_C,size, cudaMemcpyHostToDevice);
    //Launch the vec add Kernel
    int threadsperblock = 256;
    int blockspergrid = (numEle + threadsperblock - 1)/threadsperblock;
    
    vecadd<<<blockspergrid,threadsperblock>>>(d_A,d_B,d_C,numEle);
    err = cudaGetLastError();

    cudaMemcpy(h_C,d_C,size,cudaMemcpyDeviceToHost);
    //verifying
    for (int i = 0; i < numEle; ++i) {
    if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
      fprintf(stderr, "Result verification failed at element %d!\n", i);
      exit(EXIT_FAILURE);
    }
  }
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    
} 
