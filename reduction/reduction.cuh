//Generally, Bring up the Algorithm Cascading:
__global__ void stage1_reduce(float *d_in, float *d_block){
    ...
}

__global__ void stage2_reduce(float *d_block, float *d_out){
    ...
}
template <unsigned int blockSize>
__global__void reduce(const float* d_in,float* d_out,int N){

    /***************/
    // read global to shared memory
    __shared__ float sdata[blockSize];
    int tid = threadIdx.x;

    // you can do one element per thread,
    int idx = threadIdx.x + blockIdx.x*blockSize;
    sdata[tid] = idx < N? d_in[idx] : 0;


///// You can have vectorized read as well here.

    // or a while loop to add as many as necessary
    int i = blockIdx.x*(blockSize*2) + threadIdx.x;
    int gridsize = 2*blockSize*gridDim.x;
    sdata[tid] = 0;
    int sum = 0;//use a register to store the sum
    while (i < N){
        //sdata[tid] += d_in[i] + d_in[i +blockSize];
        sum += d_in[i] + d_in[i +blockSize];
        i+=gridsize;
    }
    sdata[tid] = sum;
    __syncthreads();

    if(blockSize >= 512){ 
        if(tid < 256) sdata[tid] += sdata[tid + 256];
        __syncthreads();
    }
    if(blockSize >= 256){ 
        if(tid < 128) sdata[tid] += sdata[tid + 128];
        __syncthreads();
    }
    if(blockSize >= 128){ 
        if(tid < 64) sdata[tid] += sdata[tid + 64];
        __syncthreads();
    }
    //unroll the last warp
    //put in volatile to prevent compiler optimization
    //or use war shuffle
    float val = 0.0;
    for(int mask = WARP_SIZE >> 1; mask >= 1; mask>>=1){
        val += __shfl_down_sync(0xffffffff, sdata[tid], mask);
    }
    d_out[blockIdx.x] = val;
    return val;
}

// Fully use warp shuffle : 
//绝大多数主流的 GPU 架构的 warp 都是 32 个 threads，而单个 block 最多包括 1024 个 threads，即 32 个 warps。
//即使 block_size 设置为最大的 1024，在第一次 WarpReduce 后也只有 32 个数据需要求和，只需要在第一个 warp 中继续执行 1 次WarpReduce 就可以完成 BlockReduce。
//所以实现 BlockReduce 最多需要 2 次 WarpReduce。

template <unsigned int blockSize>
__global__void reduce(const float* d_in,float* d_out,int N){

    /***************/
    int sum = 0;//use a register to store the sum
    __shared__ float sdata[32];
    int tid = threadIdx.x;
    // you can do one element per thread,
    int idx = threadIdx.x + blockIdx.x*blockSize;
    sum = idx < N? d_in[idx] : 0;


///// You can have vectorized read as well here.

    // or a while loop to add as many as necessary
    int i = blockIdx.x*(blockSize*2) + threadIdx.x;
    int gridsize = 2*blockSize*gridDim.x;
    while (i < N){
        //sdata[tid] += d_in[i] + d_in[i +blockSize];
        sum += d_in[i] + d_in[i +blockSize];//what if i + blockSize >= N?
        i+=gridsize;
    }
    //__syncthreads();
    //no need to do synthreads here, since the warp is already synchronized
    int warpId = tid/WARP_SIZE;
    int laneID = tid%WARP_SIZE;

    //in-warp reduction
    #pragma unroll
    for (int offset = WARP_SIZE>> 1; offset > 0;offset >>= 1){
        sum+=__shfl_down_sync(0xffffffff,sum,offset);
    }

    if(laneID == 0){
        sdata[warpId] = sum;
    }
    __syncthreads();
    //then do reduction for the sdata using one warp
    if(warpId == 0){
        sum = (laneID < 32) ? sdata[laneID] : 0.0f;//laneID is from  0 - 31
        #pragma unroll
        for (int offset = WARP_SIZE>> 1; offset > 0;offset >>= 1){
            sum+=__shfl_down_sync(0xffffffff,sum,offset);
        }
        if(laneID == 0){
            d_out[blockIdx.x] = sum;
        }
    }
    //d_out[blockIdx.x] = sum; //write the result to global memory
    
    //d_out Num_block


}
