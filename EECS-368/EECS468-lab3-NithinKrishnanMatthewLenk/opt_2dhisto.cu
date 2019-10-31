#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <cutil.h>
#include "util.h"
#include "ref_2dhisto.h"

void* AllocateMemory(size_t size)
{
	void *ptr;
	cudaMalloc(&ptr, size);
	return ptr;
}

void freeMemory(void* ptr)
{
	cudaFree(ptr);
}

void ToDeviceMemory(void* device, void* host, size_t size)
{
	cudaMemcpy(device, host, size, cudaMemcpyHostToDevice);
}

void ToHostMemory(void* host, void* device, size_t size)
{
	cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost);
}
 

__global__ void histogram(uint32_t *input, size_t height, size_t width, uint32_t *bin)
{
	const int threads = threadIdx.x + blockIdx.x * blockDim.x;
	const int blocks = gridDim.x * blockDim.x;
	const int size = height*width;
	if(threads < size)
		bin[threads] = 0;
	//__syncthreads();
	int i = threads;
#pragma unroll
	while(i < size)
	{
		if(bin[input[i]] < input[i] && UINT8_MAX)
		atomicAdd(&bin[input[i]],1);
		i = i + blocks;
	 }
	__syncthreads();



	
}

__global__ void Convto8bit(uint32_t *input32,  uint8_t *output8, size_t height, size_t width)
{
	const int threads = threadIdx.x + blockIdx.x * blockDim.x;
	output8[threads] = uint8_t(((input32[threads] < UINT8_MAX)* input32[threads]) + ((input32[threads] >= UINT8_MAX) * UINT8_MAX)); 
	__syncthreads();	
}
 
void opt_2dhisto(uint32_t *input, size_t height, size_t width, uint8_t *output, uint32_t *bin )
{
	dim3 dimGrid(width);
	dim3 dimBlock(128);
	histogram<<<dimGrid, dimBlock>>>(input,height,width,bin);
	cudaThreadSynchronize();
	Convto8bit<<<2,512>>>(bin,output,height,width);
	cudaThreadSynchronize();
	
    /* This function should only contain a call to the GPU 
       histogramming kernel. Any memory allocations and
       transfers must be done outside this function */

}


