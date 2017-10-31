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

#include <cuda.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

using namespace std;

#define TILE_WIDTH 32

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


__global__ void check_diag_zero(float *d_m , float *d_i , const int n)
{
	int col = threadIdx.x + (blockIdx.x*blockDim.x);	
	int row = threadIdx.y + (blockIdx.y*blockDim.y);

	if(row < n && col<n)
	{
		//Checking if diagonal element is 0
                if (d_m[(n + 1)*row] == 0)
                {
                        //checking if the row is last row. If it is last row add the previous row to make it non zero
                        if (row == (n - 1))
                        {
                                d_m[(row * n) + col] = d_m[((row - 1) * n) + col] + d_m[(row * n) + col];
                                d_i[(row * n) + col] = d_i[((row - 1) * n) + col] + d_i[(row * n) + col];
                        }
			if (row != (n-1))
                        //else    //if it is not last row, add the next row.
                        {
                                d_m[(row * n) + col] = d_m[((row + 1) * n) + col] + d_m[(row * n) + col];
                                d_i[(row * n) + col] = d_i[((row + 1) * n) + col] + d_i[(row * n) + col];
                        }
                }
	}
}

__global__ void fixRow_shared(float *d_m, float *d_I,  int n, int i)
{       
        float Aii;
        int rowId = threadIdx.x;
        
        Aii = d_m[n*i+i];
        d_m[n*i+rowId] = d_m[n*i+rowId] / Aii;
        d_I[n*i+rowId] = d_I[n*i+rowId] / Aii;
}

__global__ void fixColumn_shared(float *d_m, float *d_I, const int n, const int colId)
{
	int i = threadIdx.x;
	int j = blockIdx.x;
	float AColIdj;
	__shared__ float row[384];
	__shared__ float rowI[384];

	row[i] = d_m[colId*n+i];
	rowI[i] = d_I[colId*n+i];
	AColIdj = d_m[j*n+colId];
	__syncthreads();	

	if(i < n && j < n)
	{
		if(j != colId)
		{
			d_m[j*n+i] = d_m[j*n+i] - (AColIdj*row[i]);
			d_I[j*n+i] = d_I[j*n+i] - (AColIdj*rowI[i]);
		}
	}
}


__host__ void gpuInverseOfMatrix(float *h_matrix,float *h_iden_mat, int col)
{
	float *d_matrix,*d_iden_mat;
	const int MatSizeInBytes = col*col*sizeof(float);
	float milliseconds = 0.0f ;
	cudaError_t cudaSetDevice(int device);
        cudaSetDevice(0);
               
	cudaEvent_t kernel_start;
	cudaEvent_t kernel_stop;

	CHECK(cudaEventCreate(&kernel_start));
	CHECK(cudaEventCreate(&kernel_stop));
	CHECK(cudaEventRecord(kernel_start));
	//Allocate device memory on the global memory
	CHECK(cudaMalloc((void**)&d_matrix, MatSizeInBytes));
	CHECK(cudaMalloc((void**)&d_iden_mat, MatSizeInBytes));

	CHECK(cudaMemcpy(d_matrix, h_matrix, MatSizeInBytes, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_iden_mat, h_iden_mat, MatSizeInBytes, cudaMemcpyHostToDevice));

	//2D grid and 2D block
	int dimx1 = 32;
	int dimy1 = 32;
	dim3 block1(dimx1,dimy1);
	dim3 grid1((col+block1.x-1)/block1.x,(col+block1.y-1)/block1.y);

        int dimx2 = col;
        int dimy2 = 1;
        dim3 block2(dimx2,dimy2);                                                           
        dim3 grid2(1,1); 
	
	int dimx_fixcol = col;
        int dimy_fixcol = 1;
        dim3 block_fixcol(dimx_fixcol,dimy_fixcol);                                                           
        dim3 grid_fixcol(col,1); 


        int dimx3 = 32;
        int dimy3 = 32;
        dim3 block3(dimx3,dimy3);                                                           
	dim3 grid3((col+block3.x-1)/block3.x,(col+block3.y-1)/block3.y);

	check_diag_zero << <grid3, block3 >> >(d_matrix, d_iden_mat, col);
	for (int i = 0; i<col; i++)
	{
		fixRow_shared << <grid2, block2 >> >(d_matrix, d_iden_mat, col, i);
		fixColumn_shared << <grid_fixcol, block_fixcol >> >(d_matrix, d_iden_mat, col, i);
	}

	CHECK(cudaDeviceSynchronize());

	CHECK(cudaEventRecord(kernel_stop));
	CHECK(cudaEventSynchronize(kernel_stop));
	CHECK(cudaEventElapsedTime(&milliseconds, kernel_start, kernel_stop));
	cout << "GPU time "<<milliseconds<<endl;
	CHECK(cudaMemcpy(h_iden_mat, d_iden_mat, MatSizeInBytes, cudaMemcpyDeviceToHost));
	CHECK(cudaFree(d_matrix));
	CHECK(cudaFree(d_iden_mat));
	CHECK(cudaEventDestroy(kernel_start));
	CHECK(cudaEventDestroy(kernel_stop));
	CHECK(cudaDeviceReset());

}
