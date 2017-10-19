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

__global__ void nodiag_normalize(double *A, double *I, int n, int i){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < n && y < n)
	if (x == i && x!=y){
		I[x*n + y] /= A[i*n + i];
		A[x*n + y] /= A[i*n + i];
	}
	
}

__global__ void diag_normalize(double *A, double *I, int n, int i){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < n && y < n)
	if (x == y && x == i){
		I[x*n + y] /= A[i*n + i];
		A[x*n + y] /= A[i*n + i];
	}

}

__global__ void gaussjordan(double *A, double *I, int n, int i)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < n && y < n){
		if (x != i){
			I[x*n + y] -= I[i*n + y] * A[x*n + i];
			if (y != i){
				A[x*n + y] -= A[i*n + y] * A[x*n + i];
			}	 
		}
	}

}

__global__ void set_zero(double *A, double *I, int n, int i){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < n && y < n){
		if (x != i){
			if (y == i){
				A[x*n + y] = 0;
			}
		}
	}
}


__global__ void check_diag_zero(double *d_m , double *d_i , const int n)
{
	int col = threadIdx.x + (blockIdx.x*blockDim.x);	
	int row = threadIdx.y + (blockIdx.y*blockDim.y);

	if(row < n && col<n)
	{
	//	printf("value : %f ",d_m[(n + 1)*row]);
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


__global__ void NaiveInverse(double *d_m, double *d_i, const int n, const int i)
{
	int col = threadIdx.x + (blockIdx.x*blockDim.x);
        int row = threadIdx.y + (blockIdx.y*blockDim.y);
	if(row < n && col<n)
        {
		//Make the diagonal elements 1 along with the whole row(divide).
		double initialValue = d_m[(n + 1)*i];
		
		d_m[(i * n) + col] = d_m[(i * n) + col] / initialValue;
		d_i[(i * n) + col] = d_i[(i * n) + col] / initialValue;

		double tempDen;
		tempDen = d_m[(i * n) + i];

	
		//Making the elements of the row to zero
		double tempIni;
		tempIni = d_m[i + (col * n)]/tempDen;
		if (col != i)
		{
			for (int l = 0; l < n; l++)
			{
			d_m[(col * n)+l] = d_m[(col * n)+l] - (d_m[(i * n)+l] * tempIni);
			d_i[(col * n)+l] = d_i[(col * n)+l] - (d_i[(i * n)+l] * tempIni);
			}
		}

	}
}


__host__ void gpuInverseOfMatrix(double *h_matrix,double *h_iden_mat, int col)
{
	double *d_matrix,*d_iden_mat;
	const int MatSizeInBytes = col*col*sizeof(double);

	cudaError_t cudaSetDevice(int device);
        cudaSetDevice(0);
               
	cudaEvent_t kernel_start;
	cudaEvent_t kernel_stop;

	CHECK(cudaEventCreate(&kernel_start));
	CHECK(cudaEventCreate(&kernel_stop));

	//Allocate device memory on the global memory
	CHECK(cudaMalloc((void**)&d_matrix, MatSizeInBytes));
	CHECK(cudaMalloc((void**)&d_iden_mat, MatSizeInBytes));

	CHECK(cudaMemcpy(d_matrix, h_matrix, MatSizeInBytes, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_iden_mat, h_iden_mat, MatSizeInBytes, cudaMemcpyHostToDevice));

	//1D grid and 1D block
	int dimx = 32;
	int dimy = 32;
	dim3 block(dimx,dimy);
	dim3 grid((col+block.x-1)/block.x,(col+block.y-1)/block.y);

//	cout << "\t2D Grid Dimension" << endl;
//	cout << "\tNumber of Blocks along X dimension: " << grid.x << endl;
//	cout << "\tNumber of Blocks along Y dimension: " << grid.y << endl;
//	cout << "\t2D Block Dimension" << endl;
//	cout << "\tNumber of threads along X dimension: " << block.x << endl;
//	cout << "\tNumber of threads along Y dimension: " << block.y << endl;

	CHECK(cudaEventRecord(kernel_start));
	for (int i = 0; i<col; i++)
	{
		nodiag_normalize << <grid, block >> >(d_matrix, d_iden_mat, col, i);
		diag_normalize << <grid, block >> >(d_matrix, d_iden_mat, col, i);
		gaussjordan << <grid, block >> >(d_matrix, d_iden_mat, col, i);
		set_zero << <grid, block >> >(d_matrix, d_iden_mat, col, i);
	}

//	check_diag_zero << <grid, block >> >(d_matrix, d_iden_mat,col);
//	CHECK(cudaThreadSynchronize());
//	for(int i = 0; i < col; i++)
//	{
//		NaiveInverse << <grid, block >> >(d_matrix, d_iden_mat,col,i);		
//	}	
//	CHECK(cudaThreadSynchronize());
	CHECK(cudaEventRecord(kernel_stop));
	CHECK(cudaEventSynchronize(kernel_stop));

	CHECK(cudaMemcpy(h_matrix, d_matrix, MatSizeInBytes, cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(h_iden_mat, d_iden_mat, MatSizeInBytes, cudaMemcpyDeviceToHost));

	CHECK(cudaFree(d_matrix));
	CHECK(cudaFree(d_iden_mat));
	CHECK(cudaEventDestroy(kernel_start));
	CHECK(cudaEventDestroy(kernel_stop));
	CHECK(cudaDeviceReset());

}



