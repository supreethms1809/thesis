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

#define TILE_MN 6
#define TILE_MM 4

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

__global__ void svd_2_3_gpu(float *d_Q,float *d_M,float *d_C,float constant,int row,int col,int data_size)
{
	//int threadid = threadIdx.x;
	int blockid = blockIdx.x;
	float T = 0;
	float D = 0;
	float lam1 = 0;
	float lam2 = 0;
	float u1_norm = 0;
	float u2_norm = 0;
	//float temp_var;
	float fSum;
	int m = 2;
	int n = 3;

	__shared__ float a[TILE_MN];
	__shared__ float aat[TILE_MM];
	__shared__ float u[TILE_MM];
	__shared__ float sig[TILE_MM];
	__shared__ float vt[TILE_MN];
	__shared__ float sig_inv[TILE_MM];
	__shared__ float temp[TILE_MM];
	__shared__ float u_t[TILE_MM];
	__shared__ float Qtemp2[TILE_MN];
	//float sig_inv[m*m];
	//float temp[m*m];
	//float u_t[m*m];
	//float aat[m*m];

	for(int j = 0;j<6;j++)
	{
		a[j] = d_Q[(blockid*6)+j];
	}

		
	aat[0] = a[0]*a[0]+ a[1]*a[1] + a[2]*a[2];
	aat[1] = a[0]*a[3]+ a[1]*a[4] + a[2]*a[5];
	aat[2] = a[3]*a[0]+ a[4]*a[1] + a[5]*a[2];
	aat[3] = a[3]*a[3]+ a[4]*a[4] + a[5]*a[5]; 

	T = aat[0] + aat[3];
	D = aat[0] * aat[3] - aat[1] * aat[2];
	lam1 = 0.5*(T + (sqrt((T*T)-4*D)));
	lam2 = 0.5*(T - (sqrt((T*T)-4*D)));

	u[0] = aat[1];
	u[2] = lam1 - aat[0];
	u[1] = aat[1];
	u[3] = lam2 - aat[0];
	u1_norm =1/sqrt(u[0]*u[0]+u[2]*u[2]);
	u2_norm =1/sqrt(u[1]*u[1]+u[3]*u[3]);

	//final u
	u[0] = u[0]*u1_norm;
	u[2] = u[2]*u1_norm;
	u[1] = u[1]*u2_norm;
	u[3] = u[3]*u2_norm;

	//u_transpose
	u_t[0] = u[0];
	u_t[1] = u[2];
	u_t[2] = u[1];
	u_t[3] = u[3];
	
	//sigma 
	sig[0] = sqrt(lam1);
	sig[1] = 0;
	sig[2] = 0;
	sig[3] = sqrt(lam2);

	//sigma_inv
	sig_inv[0] = 1/sig[0];
	sig_inv[1] = 0;
	sig_inv[2] = 0;
	sig_inv[3] = 1/sig[3];

	//vt
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < m; j++)
		{
		fSum = 0.0;
			for (int k = 0; k < m; k++)
			{
			fSum += (sig_inv[(i*m) + k] * u_t[(k*m) + j]);
			}
		temp[(i*m) + j] = fSum;
		}
	}
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
		fSum = 0.0;
			for (int k = 0; k < m; k++)
			{
			fSum += (temp[(i*m) + k] * a[(k*n) + j]);
			}
		vt[(i*n) + j] = fSum;
		}
	}
	
	if((sig[0]+sig[3]) <= constant )
	{
		sig[0] = 0;
		sig[3] = 0;
	}
	else if ((sig[0] - sig[3]) <= constant)
	{
		sig[0] = ((sig[0]+sig[3])-constant)/2;
		sig[3] = sig[0];
	}
	else
	{
		sig[0] = sig[0] - constant;
		sig[3] = sig[3];
	}

	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < m; j++)
		{
		fSum = 0.0;
			for (int k = 0; k < m; k++)
			{
			fSum += (u[(i*m) + k] * sig[(k*m) + j]);
			}
		temp[(i*m) + j] = fSum;
		}
	}
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
		fSum = 0.0;
			for (int k = 0; k < m; k++)
			{
			fSum += (temp[(i*m) + k] * vt[(k*n) + j]);
			}
		Qtemp2[(i*n) + j] = fSum;
		}
	}


	//cpuMatrixMult(u,sigma,Qtemp1,ROW,ROW,ROW);
	//cpuMatrixMult(Qtemp1,vt,Qtemp2,ROW,ROW,COL);
	for(int j = 0;j<2;j++)
	{
		for(int k=0;k<3;k++)
		{
			d_M[(3 * blockid) + (j*col) + k] = Qtemp2[(j * 3) + k];
		}
	}	

	d_C[blockid] = sig[0];


}

__host__ void gpuProx_2norm(float *Q, float *M, float *C, float constant, int row, int col, int data_size)
{
	float *d_Q,*d_M,*d_C;
	const int MatSizeInBytes = row*col*sizeof(float);
	const int CsizeInBytes = data_size*sizeof(float);

	//memory allocation on GPU
	//CHECK(cudaMalloc((void**)&d_Q,MatSizeInBytes));
	//CHECK(cudaMalloc((void**)&d_M,MatSizeInBytes));
	//CHECK(cudaMalloc((void**)&d_C,CsizeInBytes));
	
	//data copy into GPU memory
	CHECK(cudaMemcpy(d_Q,Q,MatSizeInBytes,cudaMemcpyHostToDevice));

	//2D grid and 2D block
	int dimx = 1;
	int dimy = 1;
	dim3 block(dimx,dimy);
	dim3 grid(data_size,1);
//	cout << "threads in a block "<<block.x<<endl;
//	cout << "blocks in a grid "<<grid.x<<endl;

	
	svd_2_3_gpu << <grid, block >> >(d_Q, d_M, d_C, constant,row,col, data_size);

	//copy back data from GPU
	CHECK(cudaMemcpy(M,d_M,MatSizeInBytes,cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(C,d_C,CsizeInBytes,cudaMemcpyDeviceToHost));
	//CHECK(cudaFree(d_Q));
	//CHECK(cudaFree(d_M));
	//CHECK(cudaFree(d_C));


}
