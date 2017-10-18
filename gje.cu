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

	cout << "\t2D Grid Dimension" << endl;
	cout << "\tNumber of Blocks along X dimension: " << grid.x << endl;
	cout << "\tNumber of Blocks along Y dimension: " << grid.y << endl;
	cout << "\t2D Block Dimension" << endl;
	cout << "\tNumber of threads along X dimension: " << block.x << endl;
	cout << "\tNumber of threads along Y dimension: " << block.y << endl;

	CHECK(cudaEventRecord(kernel_start));
	check_diag_zero << <grid, block >> >(d_matrix, d_iden_mat,col);
	CHECK(cudaThreadSynchronize());
	for(int i = 0; i < col; i++)
	{
		NaiveInverse << <grid, block >> >(d_matrix, d_iden_mat,col,i);		
	}	
	CHECK(cudaThreadSynchronize());
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


void cpuInverseOfMatrix1(double *matrix,double *iden_mat, int col)
{
	for (int m = 0; m < col; m++)
	{
		//Checking if diagonal element is 0
		if (matrix[((col) + 1)*m] == 0)
		{
			//checking if the row is last row. If it is last row add the previous row to make it non zero
                	if (m == (col - 1))
			{
				for (int i = 0; i < (col); i++)
				{					
				matrix[(m * (col)) + i] = matrix[((m - 1) * (col)) + i] + matrix[(m * (col)) + i];
				iden_mat[(m * (col)) + i] = iden_mat[((m - 1) * (col)) + i] + iden_mat[(m * (col)) + i];
				}
			}
			else	//if it is not last row, add the next row.
			{
			        for (int i = 0; i < (col); i++)
				{
				matrix[(m * col) + i] = matrix[((m + 1) * col) + i] + matrix[(m * col) + i];
				iden_mat[(m * col) + i] = iden_mat[((m + 1) * col) + i] + iden_mat[(m * col) + i];
				}
			}
		}
	}
	for(int m=0;m<col;m++)
	{
		//Make the diagonal elements 1 along with the whole row(divide).
		double initialValue = matrix[((col) + 1)*m];
		
		for (int j = 0; j < (col); j++)
		{
		matrix[(m * (col)) + j] = matrix[(m * (col)) + j] / initialValue;
		iden_mat[(m * (col)) + j] = iden_mat[(m * (col)) + j] / initialValue;
		}

		double tempDen;
		tempDen = matrix[(m * (col)) + m];

	
		//Making the elements of the row to zero
		for (int k = 0; k < col; k++)
		{	
			double tempIni;
			tempIni = matrix[m + (k * (col))]/tempDen;
			if (k != m)
			{
				for (int l = 0; l < (col); l++)
				{
				matrix[(k * col) + l] = matrix[(k * (col)) + l] - (matrix[(m * ( col)) + l] * tempIni);
				iden_mat[(k*col)+l] = iden_mat[(k*col)+l] - (iden_mat[(m*col)+l] * tempIni);
				}
                        }

                }
        }
}

