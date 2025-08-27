template<const int NUM_THREADS = 128>
__global__ void dotproduct(float* d_a,float* d_b, float* d_out, int N){
    //each thread compute a partial sum
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    float sum = 0.0f;
    constexpr int num_warps = (NUM_THREADS + 31) / 32; //number of warps in a block
    __shared__ float sdata[num_warps]; //shared memory for partial sumsq

    int warp_id = tid / 32; //warp id
    int lane_id = tid % 32;
    //each thread compute a partial sum
    while (idx < N){
        sum += d_a[idx] * d_b[idx];
        idx += blockDim.x * gridDim.x; //stride
    }

    //use warp shuffle to reduce the partial sums
    #pragma unroll
    for(int mask = WARP_SIZE >> 1; mask > 0; mask >>= 1){
        sum += __shfl_down_sync(0xffffffff, sum, mask);
    }
    if (lane_id == 0){
        sdata[warp_id] = sum; //store the partial sum in shared memory
    }
    if(warp_id == 0){
        sum = sdata[lane_id]; //load the partial sum from shared memory
        #pragma unroll
        for (int mask = WARP_SIZE >> 1; mask > 0; mask >>= 1){
            sum += __shfl_down_sync(0xffffffff, sum, mask);
        }
        if(lane_id)
            d_out[block_id] = sum;
    }


}