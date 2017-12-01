#include <stdio.h>
#include <string>
#include <iostream>
#include <fstream>
#include <math.h>
#include <stdlib.h>
#include <limits>
#include <ctime>
#include <string>

#include <cuda.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

using namespace std;

#define CHECK(call) \
{                                                                        \
        const cudaError_t error = call;                                       \
        if (error != cudaSuccess)                                             \
        {                                                                     \
                cout << "Error: "<<__FILE__<< " : "<<__LINE__ << endl;                      \
                cout << "code: "<<error << ", reason: " <<cudaGetErrorString(error)<<endl; \
                exit(1);                                                           \
        }                                                                     \
}

#define TILE_WIDTH 16

// Compute C = A * B
__global__ void matrixMultiply(float * A, float * B, float * C, int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns) 
{
	__shared__ float ds_M[TILE_WIDTH][TILE_WIDTH];
	__shared__ float ds_N[TILE_WIDTH][TILE_WIDTH];
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x; 
	int ty = threadIdx.y;
	int Row = by * TILE_WIDTH + ty;
	int Col = bx * TILE_WIDTH + tx;
	float fSum = 0;

	for (int m = 0; m < (numAColumns-1)/TILE_WIDTH+1; ++m) 
	{
		if (Row < numARows && m*TILE_WIDTH+tx < numAColumns)
		{
			ds_M[ty][tx] = A[Row*numAColumns + m*TILE_WIDTH+tx];
		}
		else
		{
			ds_M[ty][tx] = 0;
		}

		if (Col < numBColumns && m*TILE_WIDTH+ty < numBRows)
		{
			ds_N[ty][tx] = B[(m*TILE_WIDTH+ty)*numBColumns+Col];
		}
		else
		{
			ds_N[ty][tx] = 0;
		}

		__syncthreads();
       
		for (int k = 0; k < TILE_WIDTH; ++k)
		{
			fSum += ds_M[ty][k] * ds_N[k][tx];
		}
		
		__syncthreads();
	}

	if (Row < numCRows && Col < numCColumns)
	{
		C[Row*numCColumns+Col] = fSum;
	}
}

__host__ void gpuMultShared(float *h_A, float *h_B, float *h_C, const int A_rows, const int A_cols,const int B_rows,const int B_cols)
{
	float *d_A, *d_B, *d_C;
	int C_rows,C_cols;
	const int Matrix_A_SizeInBytes = A_rows*A_cols*sizeof(float);
	const int Matrix_B_SizeInBytes = A_cols*B_cols*sizeof(float);
	const int Matrix_C_SizeInBytes = A_rows*B_cols*sizeof(float);

	C_rows = A_rows;
	C_cols = B_cols;
	cudaEvent_t kernel_start;
	cudaEvent_t kernel_stop;

	CHECK(cudaEventCreate(&kernel_start));
	CHECK(cudaEventCreate(&kernel_stop));

	//Allocate device memory on the global memory
	CHECK(cudaMalloc((void**)&d_A, Matrix_A_SizeInBytes));
	CHECK(cudaMalloc((void**)&d_B, Matrix_B_SizeInBytes));
	CHECK(cudaMalloc((void**)&d_C, Matrix_C_SizeInBytes));

	//transfer data from CPU Memory to GPU Memory
	CHECK(cudaMemcpy(d_A, h_A, Matrix_A_SizeInBytes, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_B, h_B, Matrix_B_SizeInBytes, cudaMemcpyHostToDevice));

	dim3 dimGrid((C_cols-1)/TILE_WIDTH+1, (C_rows-1)/TILE_WIDTH+1, 1);
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

	//Launch the GPU Kernel here
	CHECK(cudaEventRecord(kernel_start));
	matrixMultiply<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, A_rows, A_cols, B_rows, B_cols, C_rows, C_cols);

	cudaThreadSynchronize();
	
	CHECK(cudaEventRecord(kernel_stop));

	CHECK(cudaMemcpy(h_C, d_C, Matrix_C_SizeInBytes, cudaMemcpyDeviceToHost));

	CHECK(cudaFree(d_A));
	CHECK(cudaFree(d_B));
	CHECK(cudaFree(d_C));
	CHECK(cudaEventDestroy(kernel_start));
	CHECK(cudaEventDestroy(kernel_stop));
}


