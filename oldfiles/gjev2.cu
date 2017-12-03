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


__global__ void check_diag_zero(double *d_m , double *d_i , const int n)
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

__global__ void fixRow(double *d_m, double *d_I,  int n, int i)
{
	__shared__ double Ri[384];
	__shared__ double Ii[384];
	__shared__ double Aii;
	int colId = threadIdx.x;
	
	Ri[colId] = d_m[n*i+colId];
	Ii[colId] = d_I[n*i+colId];
	Aii = d_m[n*i+i];
	__syncthreads();
	
	Ri[colId] = Ri[colId] / Aii;
	Ii[colId] = Ii[colId] / Aii;
	d_m[n*i+colId] = Ri[colId];
	d_I[n*i+colId] = Ii[colId];

}

__global__ void fixColumn(double *d_m, double *d_I, const int n, const int colId)
{
	int i = threadIdx.x;
	int j = blockIdx.x;
	
	__shared__ double col[384];
	__shared__ double Icol[384];

	__shared__ double AColIdj;

	__shared__ double colj[384];
	__shared__ double Icolj[384];

	if(i < n && j < n)
	{
	col[i] = d_m[i*n+colId];
	Icol[i] = d_I[i*n+colId];
//	printf("threadId = %d\n",i);
//	printf("blockId = %d\n",j);
	//if(col[i] != 0)
	//{
		colj[i] = d_m[i*n+j];
		Icolj[i] = d_I[i*n+j];
		AColIdj = d_m[colId * n +j];
		//AColIdj = d_m[colId + n *j];
		__syncthreads();
		if(j != colId)
		{
		Icolj[i] = Icolj[i] - AColIdj * Icol[i];
		if(i != colId)
		{
			colj[i] = colj[i] - AColIdj * col[i];

		}
		}
		d_m[i*n+j] = colj[i];
		d_I[i*n+j] = Icolj[i];
	
	//}
	}
}

__global__ void nodiag_normalize(double *A, double *I, int n, int i)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	double temp = A[i*n + i];
	__shared__ double col[TILE_WIDTH][TILE_WIDTH];
	__shared__ double Icol[TILE_WIDTH][TILE_WIDTH];

	if (x < n && y < n)
	{
		Icol[ty][tx] = I[x*n + y];
		col[ty][tx] = A[x*n + y];
		__syncthreads();
		if (x == i && x!=y)
		{
			//I[x*n + y] /= A[i*n + i];
			//A[x*n + y] /= A[i*n + i];
//			I[x*n + y] /= temp;
//			A[x*n + y] /= temp;
			Icol[ty][tx] /= temp;
			col[ty][tx] /= temp;
		}
		I[x*n + y] = Icol[ty][tx];
		A[x*n + y] = col[ty][tx];

	}
	
}

__global__ void diag_normalize(double *A, double *I, int n, int i)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	__shared__ double col[TILE_WIDTH][TILE_WIDTH];
	__shared__ double Icol[TILE_WIDTH][TILE_WIDTH];

	double temp = A[i*n + i];
	if (x < n && y < n)
	{
		Icol[ty][tx] = I[x*n + y];
		col[ty][tx] = A[x*n + y];
		
		__syncthreads();
		if (x == y && x == i)
		{
			//I[x*n + y] /= A[i*n + i];
			//A[x*n + y] /= A[i*n + i];
//			I[x*n + y] /= temp;
//			A[x*n + y] /= temp;
			Icol[ty][tx] /= temp;
			col[ty][tx] /= temp;

		}
		I[x*n + y] = Icol[ty][tx];
		A[x*n + y] = col[ty][tx];

	}
}


__global__ void gaussjordan(double *A, double *I, int n, int i)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	__shared__ double col[TILE_WIDTH][TILE_WIDTH];
	__shared__ double Icol[TILE_WIDTH][TILE_WIDTH];
	double A_temp = A[x*n + i];
	
	if (x < n && y < n)
	{
		Icol[ty][tx] = I[x*n + y];
		col[ty][tx] = A[x*n + y];
		
		__syncthreads();
		if (x != i)
		{
//			I[x*n + y] -= I[i*n + y] * A[x*n + i];
			Icol[ty][tx] -= I[i*n + y] * A_temp;
			if (y != i)
			{
//				A[x*n + y] -= A[i*n + y] * A[x*n + i];
				col[ty][tx] -= A[i*n + y] * A_temp;
			}	 
		}
		I[x*n + y] = Icol[ty][tx];
		A[x*n + y] = col[ty][tx];
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

	//2D grid and 2D block
	int dimx1 = 32;
	int dimy1 = 32;
	dim3 block1(dimx1,dimy1);
	dim3 grid1((col+block1.x-1)/block1.x,(col+block1.y-1)/block1.y);

        int dimx2 = col;
        int dimy2 = 1;
        dim3 block2(dimx2,dimy2);                                                           
        dim3 grid2(1,1); 

        int dimx3 = 16;
        int dimy3 = 16;
        dim3 block3(dimx3,dimy3);                                                           
	dim3 grid3((col+block3.x-1)/block3.x,(col+block3.y-1)/block3.y);

//	cout << "\t2D Grid Dimension" << endl;
//	cout << "\tNumber of Blocks along X dimension: " << grid.x << endl;
//	cout << "\tNumber of Blocks along Y dimension: " << grid.y << endl;
//	cout << "\t2D Block Dimension" << endl;
//	cout << "\tNumber of threads along X dimension: " << block.x << endl;
//	cout << "\tNumber of threads along Y dimension: " << block.y << endl;

	CHECK(cudaEventRecord(kernel_start));
	check_diag_zero << <grid3, block3 >> >(d_matrix, d_iden_mat, col);
	//for (int i = 0; i<col; i++)
	//{
	//	fixRow << <grid2, block2 >> >(d_matrix, d_iden_mat, col, i);
	//	//fixColumn << <grid3, block3 >> >(d_matrix, d_iden_mat, col, i);
	//	gaussjordan << <grid3, block3 >> >(d_matrix, d_iden_mat, col, i);
	//}
	for (int i = 0; i<col; i++)
	{
		nodiag_normalize << <grid3, block3 >> >(d_matrix, d_iden_mat, col, i);
		diag_normalize << <grid3, block3 >> >(d_matrix, d_iden_mat, col, i);
		gaussjordan << <grid3, block3 >> >(d_matrix, d_iden_mat, col, i);
		//set_zero << <grid, block >> >(d_matrix, d_iden_mat, col, i);
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



