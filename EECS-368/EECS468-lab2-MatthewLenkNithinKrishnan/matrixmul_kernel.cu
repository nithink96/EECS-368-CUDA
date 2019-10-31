/*
 * Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and 
 * international Copyright laws.  
 *
 * This software and the information contained herein is PROPRIETARY and 
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and 
 * conditions of a Non-Disclosure Agreement.  Any reproduction or 
 * disclosure to any third party without the express written consent of 
 * NVIDIA is prohibited.     
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
 * OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
 * OR PERFORMANCE OF THIS SOURCE CODE.  
 *
 * U.S. Government End Users.  This source code is a "commercial item" as 
 * that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
 * "commercial computer software" and "commercial computer software 
 * documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
 * and is provided to the U.S. Government only as a commercial end item.  
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
 * source code with only those rights set forth herein.
 */

/* Matrix multiplication: C = A * B.
 * Device code.
 */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>
#include "matrixmul.h"

////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
// Matrix multiplication kernel thread specification
__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P)
{
	const int TILE_DIM = 16;

	float p_val = 0;
	

	int row = blockIdx.y* TILE_DIM + threadIdx.y;
	int col = blockIdx.x* TILE_DIM + threadIdx.x;

	__shared__ float mt [TILE_DIM+1][TILE_DIM];
	__shared__ float nt [TILE_DIM+1][TILE_DIM];

	for (int k = 0 ; k < (TILE_DIM + M.width -1)/ TILE_DIM; k++){
		
		// ensure that the data is within the scope of the matrix
		// not inside the scope when block too big for tile 
		if ( k* TILE_DIM + threadIdx.x < M.width  && row < M.height)
			mt[threadIdx.y][threadIdx.x] = M.elements[row*M.width + k * TILE_DIM + threadIdx.x];
		else
			mt[threadIdx.y][threadIdx.x] = 0.0;

		if (k*TILE_DIM + threadIdx.y < N.height && col < N.width) 
			nt[threadIdx.y][threadIdx.x] = N.elements[ (k*TILE_DIM + threadIdx.y) * N.width + col];
		else
			nt[threadIdx.y][threadIdx.x] = 0.0;

		//after everything is inserted into tile
		__syncthreads();

		// calculate the p_val
		for (int n = 0; n < TILE_DIM; ++n)
			p_val += mt[threadIdx.y][n] * nt[n][threadIdx.x];
		__syncthreads();

	}
	
	//update the P matrix with the values
	if ( row < P.height && col < P.width)
		P.elements[row*P.width+col] = p_val; 
		//P.elements[((blockIdx.y*blockDim.y+threadIdx.y)*P.width)+ (blockIdx.x*blockDim.x) + threadIdx.x] = p_val;
}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
