
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

void addWithCuda(float *c, float *a, float *b, unsigned int size);

__global__ void addKernel(float* c, float* a, float* b)
{
    // threadIdx is the local thread number 
    // blockIdx is the block number that contains the local threads
    // blockDim specifies the total number of threads in each block
    // threadIdx, blockIdx and blockDim each have multiple dimensions x, y and z
    // Use appropriate dimensions for dimension of data being used

    // Unique global index for GPU threads
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    // Pair wise addition
    c[i] = a[i] + b[i];
}

int main()
{
    const int n = 5;
    float h_a[n] = { 1.1, 2.2, 3.3, 4.4, 5.5 };
    float h_b[n] = { 10, 20, 30, 40, 50 };
    float h_c[n] = { 0 };

    // Add vectors in parallel.
    addWithCuda(h_c, h_a, h_b, n);

    printf("{1.1,2.2,3.3,4.4,5.5} + {10,20,30,40,50} = {%f,%f,%f,%f,%f}\n",
        h_c[0], h_c[1], h_c[2], h_c[3], h_c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaDeviceReset();

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
void addWithCuda(float *c, float *a, float *b, unsigned int n)
{
    // Aloccate device variables
    float *d_a = 0;
    float *d_b = 0;
    float *d_c = 0;

    int n_blocks = 1;
    int n_thds = n;

    // Size of vectors in bytes for cudamalloc
    int size = n * sizeof(float);

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaSetDevice(0);

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaMalloc((void**)&d_c, size);
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);

    // Copy input vectors from host memory to GPU buffers.
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Launch a kernel on the GPU with one thread for each element.
    // <<< number of thread blocks, number of threads in each block >>>
    addKernel<<<n_blocks, n_thds>>>(d_c, d_a, d_b);

    // Copy output vector from GPU buffer to host memory.
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_c);
    cudaFree(d_a);
    cudaFree(d_b);
}
