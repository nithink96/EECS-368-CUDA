#ifndef _PRESCAN_CU_
#define _PRESCAN_CU_

#include <assert.h>


#define NUM_BANKS 32
#define LOG_NUM_BANKS 5

#define BLOCK_SIZE 256
#define CONFLICT_FREE_OFFSET(n) ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))
#define MAX_ELEMS_PER_BLOCK 2 * BLOCK_SIZE
#define SHARED_MEMORY_SIZE MAX_ELEMS_PER_BLOCK + 30

__global__ void prescan(float *g_odata,  float * g_idata, float *sums, int n, int max_per_block)
{
 
  extern __shared__ float temp[];

  int thid = threadIdx.x;
  int offset = 1;


  int ai = thid;
  int bi = thid + blockDim.x;
  int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
  int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
  
  
  unsigned int global_index = max_per_block * blockIdx.x + threadIdx.x;

  //This deals with the last block where it may not be completely full
  if (global_index < n)
  {
    temp[ai + bankOffsetA] = g_idata[global_index];
    if (global_index + blockDim.x < n)
      temp[bi + bankOffsetB] = g_idata[global_index + blockDim.x];
  }


  //instead of going through every element, we are only looking at the max number of elements
  // possible for this block
  //build tree
  for (int d = max_per_block >> 1; d > 0; d >>= 1)
  {
    __syncthreads();

    if (thid < d)
    {
      int ai = offset * (2*thid+1) - 1;
      int bi = offset * (2*thid+ 2) - 1;
      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);
      temp[bi] += temp[ai];
    }
    offset <<= 1;
  }

  //save sum to external array
  if (thid == 0) 
  { 
    sums[blockIdx.x] = temp[max_per_block - 1 + CONFLICT_FREE_OFFSET(max_per_block - 1)];
    temp[max_per_block - 1 + CONFLICT_FREE_OFFSET(max_per_block - 1)] = 0;
  }

  //back down the tree we go
  for (int d = 1; d < max_per_block; d <<= 1)
  {
    offset >>= 1;
    __syncthreads();

    if (thid < d)
    {
      int ai = offset * (2*thid + 1) - 1;
      int bi = offset * (2*thid + 2) - 1;
      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);

      unsigned int tmp = temp[ai];
      temp[ai] = temp[bi];
      temp[bi] += tmp;
    }
  }
  __syncthreads();

  if (global_index < n)
  {
    g_odata[global_index] = temp[ai + CONFLICT_FREE_OFFSET(ai)];
    if (global_index + blockDim.x < n)
      g_odata[global_index + blockDim.x] = temp[bi + CONFLICT_FREE_OFFSET(bi)];
  }
}

__global__ void gpu_add_block_sums(float *g_odata,  float* g_idata,  float* sums,  int n)
{
  float d_block_sum_val = sums[blockIdx.x];


  int global_index = 2 * blockIdx.x * blockDim.x + threadIdx.x;

  //if the global index is valid
  if (global_index < n)
  {
    g_odata[global_index] = g_idata[global_index] + d_block_sum_val;
    if (global_index + blockDim.x < n)
      g_odata[global_index + blockDim.x] = g_idata[global_index + blockDim.x] + d_block_sum_val;
  }
}


void prescanArray_recursive(float *outArray, float *inArray, int numElements)
{

  dim3 blockDim;
  blockDim.x = BLOCK_SIZE;

  dim3 gridDim;

  int max_per_block = MAX_ELEMS_PER_BLOCK;
  gridDim.x =  numElements / max_per_block;

  //If the elements don't fit perfectly into the
  //blocks then add another block which
  // will not be filled up all of the way.

  if (numElements % max_per_block != 0) 
    gridDim.x += 1;

  int grid_size = gridDim.x;


  float* sums;
  (cudaMalloc(&sums, sizeof(float) * gridDim.x));
  (cudaMemset(sums, 0, sizeof(float) * gridDim.x));

  prescan<<<gridDim, blockDim, sizeof(float) * SHARED_MEMORY_SIZE>>>(outArray, inArray, sums, numElements, max_per_block);

  //The sums block could be greater than the number of available threads in a block
  //use recursion to keep doing the prescan and adding the blocks together until
  // they are able to fit into a single block of computation. So keep doing the
  // prescan until they fit, then add all of the blocks together at the end
  if (gridDim.x <= max_per_block)
  {
    float* d_dummy_blocks_sums;
    (cudaMalloc(&d_dummy_blocks_sums, sizeof(float)));
    (cudaMemset(d_dummy_blocks_sums, 0, sizeof(float)));
   
   prescan<<<1, blockDim, sizeof(float) * SHARED_MEMORY_SIZE>>>(sums, sums, d_dummy_blocks_sums, grid_size, max_per_block);
 
   (cudaFree(d_dummy_blocks_sums));
  }
  else
  {
    //Do recursion to restart everything
    float* inArray_tmp;
    (cudaMalloc(&inArray_tmp, sizeof(float) * grid_size));
    (cudaMemcpy(inArray_tmp, sums, sizeof(float) * grid_size, cudaMemcpyDeviceToDevice));

    prescanArray_recursive(sums, inArray_tmp, grid_size);

    (cudaFree(inArray_tmp));

  }
  
  gpu_add_block_sums<<<gridDim, blockDim>>>(outArray, outArray, sums, numElements);

  (cudaFree(sums));

}
void prescanArray(float *outArray, float *inArray, int numElements)
 {
  prescanArray_recursive(outArray, inArray, numElements);
}
#endif // _PRESCAN_CU_