#include <stdio.h>
#include <string>
#include <iostream>
#include <fstream>
#include <math.h>
#include <stdlib.h>
#include <mkl.h>
#include <mkl_lapack.h>
#include <limits>
#include <ctime>
#include <string>
#include <chrono>

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




//rows = A_rows
//cols = A_cols
__host__ void gpuMultShared(float *h_A, float *h_B, float *h_C, const int A_rows, const int A_cols,const int B_cols)
{
	float *d_A, *d_B, *d_C;
	const int Matrix_A_SizeInBytes = A_rows*A_cols*sizeof(float);
	const int Matrix_B_SizeInBytes = A_cols*B_cols*sizeof(float);
	const int Matrix_C_SizeInBytes = A_rows*B_cols*sizeof(float);

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

	//Kernel Invoke Parameters - 2D Grid and 2D Blocks
	int dimx = 32;
	int dimy = 32;

	dim3 block(dimx, dimy);
	dim3 grid((A_cols + block.x - 1) / block.x, (A_rows + block.y - 1) / block.y);

	CHECK(cudaEventRecord(kernel_start));
	SharedMult << <grid, block >> >(d_A, d_B, d_C, A_rows, A_cols,B_cols);
	CHECK(cudaEventRecord(kernel_stop));
	CHECK(cudaEventSynchronize(kernel_stop));
	CHECK(cudaEventElapsedTime(&fElapsedTime, kernel_start, kernel_stop));

	CHECK(cudaMemcpy(h_C, d_C, MatrixSizeInBytes, cudaMemcpyDeviceToHost));

	CHECK(cudaFree(d_A));
	CHECK(cudaFree(d_B));
	CHECK(cudaFree(d_C));
	CHECK(cudaEventDestroy(kernel_start));
	CHECK(cudaEventDestroy(kernel_stop));
}

#define TILE_WIDTH 32

__global__ void SharedMult(float *g_A, float *g_B, float *g_C, const int A_rows, const int A_cols, const int B_cols)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;

	//Allocate memory on the shared memory to store elements of A and B of the TILE_WIDTH x TILE_WIDTH size equal to a block
	__shared__ float s_A[TILE_WIDTH][TILE_WIDTH+2];
	__shared__ float s_B[TILE_WIDTH][TILE_WIDTH+2];
	
	//Compute gloabl row and column indexes
	int col = tx + blockDim.x * bx;
	int row = ty + blockDim.y * by;

	float fSum = 0.0f;
	for (int tw_idx = 0; tw_idx < (A_cols / TILE_WIDTH); tw_idx++) //(A_cols/TILE_WIDTH)=number of phases
	{
		//Load global elements to shared memory
		s_A[ty][tx] = g_A[(row*A_cols) + (tw_idx*TILE_WIDTH) + tx];
		s_B[ty][tx] = g_B[(tw_idx*TILE_WIDTH + ty)*A_cols + col];

		__syncthreads();

		for (int k = 0; k < TILE_WIDTH; k++)
		{
			fSum += s_A[ty][k] * s_B[k][tx];
		}
		__syncthreads();
	}
	g_C[row*A_cols + col] = fSum;
}
