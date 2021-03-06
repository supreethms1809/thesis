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

#define MAT_NUM 32
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
	int threadid = threadIdx.x;
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

	__shared__ float a[MAT_NUM][TILE_MN];
	__shared__ float aat[MAT_NUM][TILE_MM];
	__shared__ float u[MAT_NUM][TILE_MM];
	__shared__ float sig[MAT_NUM][TILE_MM];
	__shared__ float vt[MAT_NUM][TILE_MN];
	__shared__ float sig_inv[MAT_NUM][TILE_MM];
	__shared__ float temp[MAT_NUM][TILE_MM];
	__shared__ float u_t[MAT_NUM][TILE_MM];
	__shared__ float Qtemp2[MAT_NUM][TILE_MN];
	//float sig_inv[m*m];
	//float temp[m*m];
	//float u_t[m*m];
	//float aat[m*m];

	for(int j = 0;j<6;j++)
	{
		a[threadid][j] = d_Q[(blockid*192)+(threadid*6)+j];
	}

		
	aat[threadid][0] = a[threadid][0]*a[threadid][0]+ a[threadid][1]*a[threadid][1] + a[threadid][2]*a[threadid][2];
	aat[threadid][1] = a[threadid][0]*a[threadid][3]+ a[threadid][1]*a[threadid][4] + a[threadid][2]*a[threadid][5];
	aat[threadid][2] = a[threadid][3]*a[threadid][0]+ a[threadid][4]*a[threadid][1] + a[threadid][5]*a[threadid][2];
	aat[threadid][3] = a[threadid][3]*a[threadid][3]+ a[threadid][4]*a[threadid][4] + a[threadid][5]*a[threadid][5]; 

	T = aat[threadid][0] + aat[threadid][3];
	D = aat[threadid][0] * aat[threadid][3] - aat[threadid][1] * aat[threadid][2];
	lam1 = 0.5*(T + (sqrt((T*T)-4*D)));
	lam2 = 0.5*(T - (sqrt((T*T)-4*D)));

	u[threadid][0] = aat[threadid][1];
	u[threadid][2] = lam1 - aat[threadid][0];
	u[threadid][1] = aat[threadid][1];
	u[threadid][3] = lam2 - aat[threadid][0];
	u1_norm =1/sqrt(u[threadid][0]*u[threadid][0]+u[threadid][2]*u[threadid][2]);
	u2_norm =1/sqrt(u[threadid][1]*u[threadid][1]+u[threadid][3]*u[threadid][3]);

	//final u
	u[threadid][0] = u[threadid][0]*u1_norm;
	u[threadid][2] = u[threadid][2]*u1_norm;
	u[threadid][1] = u[threadid][1]*u2_norm;
	u[threadid][3] = u[threadid][3]*u2_norm;

	//u_transpose
	u_t[threadid][0] = u[threadid][0];
	u_t[threadid][1] = u[threadid][2];
	u_t[threadid][2] = u[threadid][1];
	u_t[threadid][3] = u[threadid][3];
	
	//sigma 
	sig[threadid][0] = sqrt(lam1);
	sig[threadid][1] = 0;
	sig[threadid][2] = 0;
	sig[threadid][3] = sqrt(lam2);

	//sigma_inv
	sig_inv[threadid][0] = 1/sig[threadid][0];
	sig_inv[threadid][1] = 0;
	sig_inv[threadid][2] = 0;
	sig_inv[threadid][3] = 1/sig[threadid][3];

	//vt
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < m; j++)
		{
		fSum = 0.0;
			for (int k = 0; k < m; k++)
			{
			fSum += (sig_inv[threadid][(i*m) + k] * u_t[threadid][(k*m) + j]);
			}
		temp[threadid][(i*m) + j] = fSum;
		}
	}
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
		fSum = 0.0;
			for (int k = 0; k < m; k++)
			{
			fSum += (temp[threadid][(i*m) + k] * a[threadid][(k*n) + j]);
			}
		vt[threadid][(i*n) + j] = fSum;
		}
	}
	
	if((sig[threadid][0]+sig[threadid][3]) <= constant )
	{
		sig[threadid][0] = 0;
		sig[threadid][3] = 0;
	}
	else if ((sig[threadid][0] - sig[threadid][3]) <= constant)
	{
		sig[threadid][0] = ((sig[threadid][0]+sig[threadid][3])-constant)/2;
		sig[threadid][3] = sig[threadid][0];
	}
	else
	{
		sig[threadid][0] = sig[threadid][0] - constant;
		sig[threadid][3] = sig[threadid][3];
	}

	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < m; j++)
		{
		fSum = 0.0;
			for (int k = 0; k < m; k++)
			{
			fSum += (u[threadid][(i*m) + k] * sig[threadid][(k*m) + j]);
			}
		temp[threadid][(i*m) + j] = fSum;
		}
	}
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
		fSum = 0.0;
			for (int k = 0; k < m; k++)
			{
			fSum += (temp[threadid][(i*m) + k] * vt[threadid][(k*n) + j]);
			}
		Qtemp2[threadid][(i*n) + j] = fSum;
		}
	}


	//cpuMatrixMult(u,sigma,Qtemp1,ROW,ROW,ROW);
	//cpuMatrixMult(Qtemp1,vt,Qtemp2,ROW,ROW,COL);
	for(int j = 0;j<2;j++)
	{
		for(int k=0;k<3;k++)
		{
			d_M[(3 * ((blockid*MAT_NUM) + threadid)) + (j*col) + k] = Qtemp2[threadid][(j * 3) + k];
		}
	}	

	d_C[(blockid*MAT_NUM) + threadid] = sig[threadid][0];


}

__host__ void gpuProx_2norm(float *Q, float *M, float *C, float constant, int row, int col, int data_size)
{
	float *d_Q,*d_M,*d_C;
	const int MatSizeInBytes = row*col*sizeof(float);
	const int CsizeInBytes = data_size*sizeof(float);

	//memory allocation on GPU
	CHECK(cudaMalloc((void**)&d_Q,MatSizeInBytes));
	CHECK(cudaMalloc((void**)&d_M,MatSizeInBytes));
	CHECK(cudaMalloc((void**)&d_C,CsizeInBytes));
	
	//data copy into GPU memory
	CHECK(cudaMemcpy(d_Q,Q,MatSizeInBytes,cudaMemcpyHostToDevice));

	//2D grid and 2D block
	int dimx = 32;
	int dimy = 1;
	dim3 block(dimx,dimy);
	dim3 grid(data_size/32,1);
//	cout << "threads in a block "<<block.x<<endl;
//	cout << "blocks in a grid "<<grid.x<<endl;

	
	svd_2_3_gpu << <grid, block >> >(d_Q, d_M, d_C, constant,row,col, data_size);

	//copy back data from GPU
	CHECK(cudaMemcpy(M,d_M,MatSizeInBytes,cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(C,d_C,CsizeInBytes,cudaMemcpyDeviceToHost));
	CHECK(cudaFree(d_Q));
	CHECK(cudaFree(d_M));
	CHECK(cudaFree(d_C));


}
