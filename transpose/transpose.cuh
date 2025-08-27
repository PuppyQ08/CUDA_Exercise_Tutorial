
//naive
#define COORD(x,y,width) ((y)*width + (x))
__global__ void transpose(float *d_in,float* d_out, int M, int N){
    //BlockIDX x and y is in cartesian coordinate style
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < N && y < M){
        d_out[x*M + y] = d_in[y*N + x];
    }
}
//naive matreix transpose kernel
__global__ void transpose(float* d_in, float* d_out, int M, int N){
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM; 
    // we loop in the y direction to get coalesced memeory access
    for(int j = 0; j < TILE_DIM; j += BLOCK_ROWS){
        d_out[x*width + (y+j)] = d_in[(y+j)*width + x];
    }
}

//matrix transpose via shared memory
__global__ void transpose(float* d_in, float* d_out, int M, int N){
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    __shared__ float tile[TILE_DIM][TILE_DIM+1]; //avoid bank conflict
    int width = gridDim.x * TILE_DIM; 
    // we loop in the y direction to get coalesced memeory access
    for(int j = 0; j < TILE_DIM; j += BLOCK_ROWS){
        tile[threadIdx.y + j][threadIdx.x] = d_in[(y+j)*width+x];
    }//direct copy
    __syncthreads();
    x = blockIdx.y * TILE_DIM + threadIdx.x; //transpose block offset
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    for(int j = 0; j < TILE_DIM; j += BLOCK_ROWS){
        d_out[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
    }
}
