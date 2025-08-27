#include <cuda_runtime.h>
#define FLOAT4(ptr) (reinterpret_cast<float4*>(&(ptr))[0])
/******************/
//C style :
// #define FLOAT4(ptr) *(float4*)(&(ptr))
/*****************/
#define CEIL(a, b) ((a+b-1)/b)
//we just write the optimized kernel here
//XXX Use float4 to do vectorized memory access
//对于访存单元，在读取 4 个 float 数据时，会发送 4 条 LD.E 指令。向量化访存则是通过一条 LD.E.128 直接读取 4 个 float 数据，对应 CUDA 中的 float4 数据类型。相比之下，向量化访存减少了总的指令数，降低了延迟，提高了带宽的利用
//当然，转成float4也会产生一些负面的影响，首先是所采用的寄存器更多了，寄存器资源被占用多了之后，SM中能够并发的warp数量会有所减少。
//此外，如果本身程序的并行粒度就不太够，使用float4的话，所使用的block数量减少，warp数量减少，性能也会有一定的影响。所以如果是并行粒度本身不太够的情况下，还是需要谨慎地考虑是否采用float4这样的向量化数据。

__global__ void vecadd(const flaot *A, const float *B, float *C, int numEle){
    int  idx = 4*(blockDim.x*blockIdx.x + threadIdx.x);
    int stride = 4*blockDim.x*gridDim.x;
    for(int i = idx;i< numEle;i+=stride){
        if (idx < numEle - 4 ) {
            //C[idx] = A[idx] + B[idx];
            float4 reg_a = FLOAT4(A[idx]);
            float4 reg_b = FLOAT4(B[idx]);
            float4 reg_c;
            reg_c.x = reg_a.x + reg_b.x;
            reg_c.y = reg_a.y + reg_b.y;
            reg_c.z = reg_a.z + reg_b.z;
            reg_c.w = reg_a.w + reg_b.w;
            FLOAT4(C[idx]) = reg_c;
        }else{
            #pragma unroll
            for(int j = idx;j < numEle;j++){
                C[j] = A[j] + B[j];
            }
        }
    }
}
int main(){
    int blockSize = 1024; //could be adjusted to be like 256
    //******************
    //For block size: 
    // Bsize > Max threads per SM / Max blocks per SM. otherwise it can't reach the max occupancy.
    // Bsize can't be too large, othewise the register usage reach limit.
    // Grid size shoule be large enough to keep all SMs busy. 
    // Grid size should be multiply of wave size.
    
    //int gridSize = (N + blockSize - 1) / blockSize;
    int grirdSize = CEIL(N, blockSize*4); //because each thread process 4 elements
    //notice you need to change the grid size accordingly, not the block size, otherwise it will reduce the Occupancy in SM.
    }