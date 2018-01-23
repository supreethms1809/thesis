#include <stdio.h>
#include <string>
#include <iostream>
#include <fstream>
#include <math.h>
#include <stdlib.h>
#include <limits>
#include <ctime>
#include <string>
#include <cublas_v2.h>

#include <cuda.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

using namespace std;

#define TILE_DIM 32;
#define BLOCK_ROWS 8;
#define TILE_MN 6
#define TILE_MM 4
#define TILE_WIDTH 16
#define TILE_WIDTH_INVERSE 1536
__device__ float d_prim ;
__device__ float d_dual ;
__device__ float d_mu;
//__device__ int d_flag;
__device__ float d_tol;
__device__ float MminusZ_norm;
__device__ float MminusZ_sum;
__device__ float ZminusZ0_norm;
__device__ float ZminusZ0_sum;
__device__ float Z0_norm;
__device__ float Z0_sum;

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


__global__ void transposeOnGPU(float *d_B, float *d_Bt, int rows, int cols,float mu)
{
	unsigned int iy = threadIdx.x + (blockIdx.x*blockDim.x);
	unsigned int ix = threadIdx.y + (blockIdx.y*blockDim.y);
	if(iy==1&&ix==1)
	{
		d_mu = mu;
		d_tol = 1e-04;
	//	printf("value of d_mu: %f \n",d_mu);
	}

	if (ix < rows && iy < cols)
	{
		d_Bt[iy*rows + ix] = d_B[ix*cols + iy];
	}
}

__global__ void VecMatMulrow(float *d_B,float *d_mui_B ,int m,int n)
{
	int col = threadIdx.x + blockIdx.x * blockDim.x; 	
	int row = threadIdx.y + blockIdx.y * blockDim.y; 
	float muinv = 1/d_mu;
	
	if(col < n && row < m)
	{
		d_mui_B[row*n+col] = muinv*d_B[row*n+col];	
	}	
}

__global__ void MatVecMulrow(float *d_Bt,float *d_Bt_mui ,int m,int n)
{
	int col = threadIdx.x + blockIdx.x * blockDim.x; 	
	int row = threadIdx.y + blockIdx.y * blockDim.y; 
	float muinv = 1/d_mu;
	
//	if(col==0&&row==0)
//	printf("value of muinv: %f \n",muinv);
	if(col < n && row < m)
	{
		d_Bt_mui[row*n+col] = d_Bt[row*n+col]*muinv;	
	}	
}

__global__ void InnerMultiplication(float * A, float * B, float * C, int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns) 
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
		if (Row == Col)
		{
			C[Row*numCColumns+Col] = fSum + 1;
		}
		else
		{
			C[Row*numCColumns+Col] = fSum;
		}
	}
}

__global__ void OuterMultiplication1(float * A, float * B, float * C, int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns) 
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

__global__ void OuterMultiplication2_sub(float * A, float * B, float * C, int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns) 
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
	float muinv = 1/d_mu;

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
		if (Row == Col)
		{
			C[Row*numCColumns+Col] = muinv - fSum;
		}
		else
		{
			C[Row*numCColumns+Col] = 0.0f - fSum;
		}
	}
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
	__shared__ float row[TILE_WIDTH_INVERSE];
	__shared__ float rowI[TILE_WIDTH_INVERSE];

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

__global__ void addmu_diagonal(float *d_bbt,float *d_Zden_bbt,float mu,int row, int col)
{
	unsigned int iy = threadIdx.x + (blockIdx.x*blockDim.x);
	unsigned int ix = threadIdx.y + (blockIdx.y*blockDim.y);

	if(ix==0 && iy==0)
	{
		d_mu = mu;
		d_tol = 1e-04;
	}

	if (ix < row && iy < col)
	{
		if(ix==iy)
		{
			d_Zden_bbt[ix*col + iy] = d_bbt[ix*col + iy]+mu;
		}
		if(ix!=iy)
		{
			d_Zden_bbt[ix*col + iy] = d_bbt[ix*col + iy];
		}
	}
}

__global__ void addmu_diagonalgpumu(float *d_bbt,float *d_Zden_bbt,int row, int col)
{
	unsigned int iy = threadIdx.x + (blockIdx.x*blockDim.x);
	unsigned int ix = threadIdx.y + (blockIdx.y*blockDim.y);

	if (ix < row && iy < col)
	{
		if(ix==iy)
		{
			d_Zden_bbt[ix*col + iy] = d_bbt[ix*col + iy]+d_mu;
		}
		if(ix!=iy)
		{
			d_Zden_bbt[ix*col + iy] = d_bbt[ix*col + iy];
		}
	}
}



__global__ void eye_gpu(float *d_Zden,int row, int col)
{
	unsigned int iy = threadIdx.x + (blockIdx.x*blockDim.x);
	unsigned int ix = threadIdx.y + (blockIdx.y*blockDim.y);

	if (ix < row && iy < col)
	{
		if(ix==iy)
		{
			d_Zden[ix*col + iy] = 1.0f;
		}
		if(ix!=iy)
		{
			d_Zden[ix*col + iy] = 0.0f;
		}
	}

}



__global__ void initializeZGPU(float *d_Z, float *d_Z0, int row, int col)
{
	unsigned int iy = threadIdx.x + (blockIdx.x*blockDim.x);
	unsigned int ix = threadIdx.y + (blockIdx.y*blockDim.y);

	if (ix < row && iy < col)
	{
		d_Z0[ix*col + iy] = d_Z[ix*col + iy];
	}
}

__global__ void tempnum1Calc(float *d_tempNum1,float *d_xy,float *d_T,int row,int col)
{
	unsigned int iy = threadIdx.x + (blockIdx.x*blockDim.x);
	unsigned int ix = threadIdx.y + (blockIdx.y*blockDim.y);

	if (ix < row && iy < col)
	{
		d_tempNum1[ix*col + iy] = d_xy[ix*col + iy]-d_T[ix];
	}
}

__global__ void sumOfMatrixGPU(float *d_Znum,float *d_tempNum2,float *d_M,float *d_Y,int row,int col)
{
	unsigned int iy = threadIdx.x + (blockIdx.x*blockDim.x);
	unsigned int ix = threadIdx.y + (blockIdx.y*blockDim.y);

	if (ix < row && iy < col)
	{
		d_Znum[ix*col + iy] = d_tempNum2[ix*col + iy]+d_mu*d_M[ix*col + iy]+d_Y[ix*col + iy];
	}
}

__global__ void calculateQGPU(float *d_Q,float *d_Z,float *d_Y,int row,int col)
{
	float oneovermu;
	oneovermu = 1/d_mu;
	unsigned int iy = threadIdx.x + (blockIdx.x*blockDim.x);
	unsigned int ix = threadIdx.y + (blockIdx.y*blockDim.y);

	if (ix < row && iy < col)
	{
		d_Q[ix*col + iy] = d_Z[ix*col + iy] - (oneovermu*d_Y[ix*col + iy]);
	}

}

__global__ void reorderQ(float *d_Q,float *d_Q_re,int row,int col,int data_size)
{
	int blockid = blockIdx.x;

	for(int j = 0;j<2;j++)
	{
		for(int k=0;k<3;k++)
		{
			d_Q_re[(6 * blockid) + (j*3) + k] = d_Q[(3*blockid)+(j * col) + k];
		}
	}	
}

__global__ void svd_2_3_gpu(float *d_Q,float *d_M,float *d_C,float lam,int row,int col,int data_size)
{
	//int threadid = threadIdx.x;
	int blockid = blockIdx.x;
	float constant = lam/d_mu;
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


	for(int j = 0;j<2;j++)
	{
		for(int k=0;k<3;k++)
		{
			d_M[(3 * blockid) + (j*col) + k] = Qtemp2[(j * 3) + k];
		}
	}	

	d_C[blockid] = sig[0];


}

__global__ void updateYGPU(float *d_Y,float *d_M,float *d_Z,int row,int col)
{
	unsigned int iy = threadIdx.x + (blockIdx.x*blockDim.x);
	unsigned int ix = threadIdx.y + (blockIdx.y*blockDim.y);

	if (ix < row && iy < col)
	{
		d_Y[ix*col + iy] += d_mu*(d_M[ix*col + iy] - d_Z[ix*col + iy]);
	}
}

__device__ void norm1()
{
	MminusZ_norm = sqrtf(MminusZ_sum);
	ZminusZ0_norm = sqrtf(ZminusZ0_sum);
	Z0_norm = sqrtf(Z0_sum);

	d_prim = (MminusZ_norm/Z0_norm);		
//	printf("MminusZ_norm %f \t",MminusZ_norm);
//	printf("Z0_norm %f \t",Z0_norm);
	d_dual = (d_mu*ZminusZ0_norm/Z0_norm);
//	printf("ZminusZ0_norm %f \t",ZminusZ0_norm);
//	printf("d_prim %f \t",d_prim);
//	printf("d_dual %f \t",d_dual);
//	printf("d_mu %f \n",d_mu);


}

__device__ void checkcondition(int *d_flag)
{
	if((d_prim<d_tol)&&(d_dual<d_tol))
	{
		*d_flag = 2;
	}	
	else
	{
		if(d_prim > (10*d_dual))
		{
			d_mu = 2*d_mu;
			*d_flag = 1;
		}
		else if(d_dual > (10*d_prim))
		{
			d_mu = d_mu/2;
			*d_flag = 1;
		}
		else
		{
			*d_flag = 0;
		}
	}


}

__global__ void resCalcGPU(float *d_M,float *d_Z,float *d_Z0,float *d_MminusZ,float *d_ZminusZ0,int row,int col,int *d_flag)
{
	
	unsigned int iy = threadIdx.x + (blockIdx.x*blockDim.x);
	unsigned int ix = threadIdx.y + (blockIdx.y*blockDim.y);

	MminusZ_norm = 0.0f;
	MminusZ_sum = 0.0f;
	ZminusZ0_norm = 0.0f;
	ZminusZ0_sum = 0.0f;
	Z0_norm = 0.0f;
	Z0_sum = 0.0f;


	if (ix < row && iy < col)
	{
		d_MminusZ[ix*col + iy] = d_M[ix*col + iy] - d_Z[ix*col + iy];
		d_ZminusZ0[ix*col + iy] = d_Z[ix*col + iy] - d_Z0[ix*col + iy];
	}

	if (ix < row && iy < col)
	{
		atomicAdd(&MminusZ_sum, (d_MminusZ[ix*col + iy] * d_MminusZ[ix*col + iy]));
		atomicAdd(&ZminusZ0_sum, (d_ZminusZ0[ix*col + iy] * d_ZminusZ0[ix*col + iy]));
		atomicAdd(&Z0_sum, (d_Z0[ix*col + iy] * d_Z0[ix*col + iy]));
	}


	if(ix==1 && iy==1)
	{
	norm1();
	checkcondition(d_flag);
	}

}


__host__ void loop(float *xy,float *B,float *Bt,float *Zden,float *Z,float *Z0,float *T,float *M,float *Y,float *Q,float *Q_re,float *C,float *prim, float *dual,int row,int col,int row1,int col1,float mu,float lam,int data_size)
{
	float *d_xy;
	float *d_B;
	float *d_Bt;
	float *d_bbt;
	float *d_Zden;
	float *d_Z;
	float *d_Z0;
	float *d_Zden_bbt;
	float *d_T;
	float *d_tempNum1;
	float *d_tempNum2;
	float *d_Znum;
	float *d_M;
	float *d_Y;
	float *d_Q;
	float *d_Q_re;
	float *d_C;
	float *d_MminusZ;
	float *d_ZminusZ0;
	float *d_mui_B;
	float *d_Bt_mui;
	float *d_temp_inner;
	float *d_temp_inner_inv;
	float *d_temp1;

	float MminusZ_norm = 0.0f;
	float ZminusZ0_norm = 0.0f;
	float Z0_norm = 0.0f;
	float tol = 1e-04;
	int elements = 0;
	elements = row*row1;
	int flag = 0;
	int * d_flag;
	
	cublasHandle_t handle;
	cudaEvent_t kernel_start;
	cudaEvent_t kernel_stop;

	CHECK(cudaEventCreate(&kernel_start));
	CHECK(cudaEventCreate(&kernel_stop));

	const int xy_size = row*col*sizeof(float);
	const int B_size = row1*col1*sizeof(float);
	const int Bt_size = col1*row1*sizeof(float);
	const int bbt_size = row1*row1*sizeof(float);
	const int Zden_size = row1*row1*sizeof(float);
	const int Z_size = row*row1*sizeof(float);
	const int T_size = row*sizeof(float);
	const int tempNum1_size = row*col*sizeof(float);
	const int tempNum2_size = row*row1*sizeof(float);
	const int Znum_size = row*row1*sizeof(float);
	const int M_size = row*row1*sizeof(float);
	const int Y_size = row*row1*sizeof(float);
	const int Q_size = row*row1*sizeof(float);
	const int element_size = 1*sizeof(float);
	const int mui_size = row1*sizeof(float);
	const int innerinv_size = col*col*sizeof(float);


	CHECK(cudaMalloc((void**)&d_xy,xy_size));
	CHECK(cudaMalloc((void**)&d_B,B_size));
	CHECK(cudaMalloc((void**)&d_Bt,Bt_size));
	CHECK(cudaMalloc((void**)&d_bbt,bbt_size));
	CHECK(cudaMalloc((void**)&d_Zden_bbt,bbt_size));
	CHECK(cudaMalloc((void**)&d_Zden,Zden_size));
	CHECK(cudaMalloc((void**)&d_Z,Z_size));
	CHECK(cudaMalloc((void**)&d_Z0,Z_size));
	CHECK(cudaMalloc((void**)&d_T,T_size));
	CHECK(cudaMalloc((void**)&d_tempNum1,tempNum1_size));
	CHECK(cudaMalloc((void**)&d_tempNum2,tempNum2_size));
	CHECK(cudaMalloc((void**)&d_Znum,Znum_size));
	CHECK(cudaMalloc((void**)&d_M,M_size));
	CHECK(cudaMalloc((void**)&d_Y,Y_size));
	CHECK(cudaMalloc((void**)&d_Q,Q_size));
	CHECK(cudaMalloc((void**)&d_Q_re,Q_size));
	CHECK(cudaMalloc((void**)&d_C,data_size));
	CHECK(cudaMalloc((void**)&d_MminusZ,Q_size));
	CHECK(cudaMalloc((void**)&d_ZminusZ0,Q_size));
	CHECK(cudaMalloc((void**)&d_flag, sizeof(int)));

	CHECK(cudaMalloc((void**)&d_mui_B,B_size));
	CHECK(cudaMalloc((void**)&d_Bt_mui,Bt_size));
	CHECK(cudaMalloc((void**)&d_temp_inner,innerinv_size));
	CHECK(cudaMalloc((void**)&d_temp_inner_inv,innerinv_size));
	CHECK(cudaMalloc((void**)&d_temp1,Bt_size));

	CHECK(cudaMemcpy(d_xy,xy,xy_size,cudaMemcpyHostToDevice));	
	CHECK(cudaMemcpy(d_B,B,B_size,cudaMemcpyHostToDevice));	
	CHECK(cudaMemcpy(d_flag, &flag, sizeof(int), cudaMemcpyHostToDevice));

	int dimx_transpose = 16;
	int dimy_transpose = 16;
	dim3 block_transpose(dimx_transpose,dimy_transpose);
	dim3 grid_transpose((col1+block_transpose.x-1)/block_transpose.x,(row1+block_transpose.y-1)/block_transpose.y);

	int dimx_bbt = 16;
	int dimy_bbt = 16;
	dim3 block_bbt(dimx_bbt,dimy_bbt);
	dim3 grid_bbt((row1+block_bbt.x-1)/block_bbt.x,(row1+block_bbt.y-1)/block_bbt.y);
	
	int dimx_Zcalc_muiB = 16;
	int dimy_Zcalc_muiB = 16;
	dim3 block_Zcalc_muiB(dimx_Zcalc_muiB,dimy_Zcalc_muiB);
	dim3 grid_Zcalc_muiB((col+block_Zcalc_muiB.x-1)/block_Zcalc_muiB.x,(row1+block_Zcalc_muiB.y-1)/block_Zcalc_muiB.y);

	int dimx_Zcalc_Btmui = 16;
	int dimy_Zcalc_Btmui = 16;
	dim3 block_Zcalc_Btmui(dimx_Zcalc_Btmui,dimy_Zcalc_Btmui);
	dim3 grid_Zcalc_Btmui((row1+block_Zcalc_Btmui.x-1)/block_Zcalc_Btmui.x,(col+block_Zcalc_Btmui.y-1)/block_Zcalc_Btmui.y);

	int dimx_wood_inner_mult = 16;
	int dimy_wood_inner_mult = 16;
	dim3 block_wood_inner_mult(dimx_wood_inner_mult,dimy_wood_inner_mult);
	dim3 grid_wood_inner_mult((col+block_wood_inner_mult.x-1)/block_wood_inner_mult.x,(col+block_wood_inner_mult.y-1)/block_wood_inner_mult.y);


	//2D grid and 2D block
        int dimx2 = col;
        int dimy2 = 1;
        dim3 block2(dimx2,dimy2);   
        dim3 grid2(1,1); 
	
	int dimx_fixcol = col;
        int dimy_fixcol = 1;
        dim3 block_fixcol(dimx_fixcol,dimy_fixcol);     
	dim3 grid_fixcol(col,1); 

        int dimx3 = 16;
        int dimy3 = 16;
        dim3 block3(dimx3,dimy3); 
	dim3 grid3((col+block3.x-1)/block3.x,(col+block3.y-1)/block3.y);

        int dimx_wood_inv = 16;
        int dimy_wood_inv = 16;
        dim3 block_wood_inv(dimx_wood_inv,dimy_wood_inv); 
	dim3 grid_wood_inv((row1+block_wood_inv.x-1)/block_wood_inv.x,(row1+block_wood_inv.y-1)/block_wood_inv.y);


	int dimx_ini = 16;
	int dimy_ini = 16;
	dim3 block_ini(dimx_ini,dimy_ini);
	dim3 grid_ini((row1+block_ini.x-1)/block_ini.x,(row+block_ini.y-1)/block_ini.y);

	int dimx_tempnum1 = 16;
	int dimy_tempnum1 = 16;
	dim3 block_tempnum1(dimx_tempnum1,dimy_tempnum1);
	dim3 grid_tempnum1((col+block_ini.x-1)/block_ini.x,(row+block_ini.y-1)/block_ini.y);

	int dimx_re = 1;
	int dimy_re = 1;
	dim3 block_re(dimx_re,dimy_re);
	dim3 grid_re(data_size,1);

	CHECK(cudaEventRecord(kernel_start));
	

	transposeOnGPU << <grid_transpose, block_transpose >> >(d_B, d_Bt, row1, col1,mu);
//	cudaThreadSynchronize();

	VecMatMulrow<< <grid_Zcalc_muiB,block_Zcalc_muiB >> >(d_B,d_mui_B , row1, col);
	MatVecMulrow<< <grid_Zcalc_Btmui,block_Zcalc_Btmui >> >(d_Bt,d_Bt_mui ,col,row1);
	InnerMultiplication<< <grid_wood_inner_mult,block_wood_inner_mult >> >(d_Bt, d_mui_B, d_temp_inner, col, row1, row1, col, col, col); 
	eye_gpu<<<grid_bbt,block_bbt>>>(d_temp_inner_inv,col, col);
	
	check_diag_zero << <grid3, block3 >> >(d_temp_inner, d_temp_inner_inv, col);
	for (int i = 0; i<col; i++)
	{
		fixRow_shared << <grid2, block2 >> >(d_temp_inner, d_temp_inner_inv, col, i);
		fixColumn_shared << <grid_fixcol, block_fixcol >> >(d_temp_inner, d_temp_inner_inv, col, i);
	}
	
	OuterMultiplication1<<<grid_Zcalc_Btmui,block_Zcalc_Btmui >>>(d_temp_inner_inv, d_Bt_mui, d_temp1, col, col, col, row1, col, row1); 
	OuterMultiplication2_sub<<<grid_wood_inv,block_wood_inv >>>(d_mui_B, d_temp1, d_Zden, row1, col, col, row1, row1, row1); 
//	cudaThreadSynchronize();
	
	for(int iter=0;iter<500;iter++)
	{
		initializeZGPU<< <grid_ini,block_ini >> >(d_Z,d_Z0,row,row1);

		if(flag == 1)
		{
		VecMatMulrow<< <grid_Zcalc_muiB,block_Zcalc_muiB >> >(d_B,d_mui_B , row1, col);
		MatVecMulrow<< <grid_Zcalc_Btmui,block_Zcalc_Btmui >> >(d_Bt,d_Bt_mui ,col,row1);
		InnerMultiplication<< <grid_wood_inner_mult,block_wood_inner_mult >> >(d_Bt, d_mui_B, d_temp_inner, col, row1, row1, col, col, col); 
		eye_gpu<<<grid_bbt,block_bbt>>>(d_temp_inner_inv,col, col);
	
		check_diag_zero << <grid3, block3 >> >(d_temp_inner, d_temp_inner_inv, col);
		for (int i = 0; i<col; i++)
		{
			fixRow_shared << <grid2, block2 >> >(d_temp_inner, d_temp_inner_inv, col, i);
			fixColumn_shared << <grid_fixcol, block_fixcol >> >(d_temp_inner, d_temp_inner_inv, col, i);
		}
	
		OuterMultiplication1<<<grid_Zcalc_Btmui,block_Zcalc_Btmui >>>(d_temp_inner_inv, d_Bt_mui, d_temp1, col, col, col, row1, col, row1); 
		OuterMultiplication2_sub<<<grid_wood_inv,block_wood_inv >>>(d_mui_B, d_temp1, d_Zden, row1, col, col, row1, row1, row1); 

		}

		//Z calculation	
		//calculateZ_preZden(Z, Zden_inv,xy, E, T, B_transpose,mu,M,Y,row,col,row1);
		tempnum1Calc<<<grid_tempnum1,block_tempnum1>>>(d_tempNum1,d_xy,d_T,row,col);
//		cudaThreadSynchronize();
		matrixMultiply<<<grid_ini,block_ini>>>(d_tempNum1,d_Bt,d_tempNum2,row,col,col1,row1,row,row1);
//		cudaThreadSynchronize();
		sumOfMatrixGPU<<<grid_ini,block_ini>>>(d_Znum,d_tempNum2, d_M, d_Y, row, row1);
//		cudaThreadSynchronize();
		matrixMultiply<<<grid_ini,block_ini>>>(d_Znum,d_Zden,d_Z,row,row1,row1,row1,row,row1);
//		cudaThreadSynchronize();

		//calculateQ(Q,Q_re,Z,Y,mu,row,row1,data_size);
		calculateQGPU<<<grid_ini,block_ini>>>(d_Q,d_Z,d_Y,row,row1);
//		cudaThreadSynchronize();
		reorderQ<<<grid_re,block_re>>>(d_Q,d_Q_re,row,row1,data_size);	
//		cudaThreadSynchronize();

		//prox_norm calculation
		svd_2_3_gpu << <grid_re, block_re >> >(d_Q_re, d_M, d_C, lam,row,row1,data_size);
//		cudaThreadSynchronize();

		//updateDualvariable
		updateYGPU<<<grid_ini,block_ini>>>(d_Y,d_M,d_Z,row,row1);
//		cudaThreadSynchronize();
		
		//residual calculation
		resCalcGPU<<<grid_ini,block_ini>>>(d_M,d_Z,d_Z0,d_MminusZ,d_ZminusZ0,row,row1,d_flag);	
//		cudaThreadSynchronize();

//		cout << "iter = "<<iter+1<<endl;
		CHECK(cudaMemcpy( &flag,d_flag, sizeof(int), cudaMemcpyDeviceToHost));
		if(flag == 2)
                {
			cout << "iter = "<<iter+1<<endl;
			break;
		}
	}

	cudaThreadSynchronize();
	
	CHECK(cudaEventRecord(kernel_stop));
	CHECK(cudaEventSynchronize(kernel_stop));
	float milliseconds = 0;
	CHECK(cudaEventElapsedTime(&milliseconds, kernel_start, kernel_stop));
	cout << "Time in ms : "<<milliseconds << " ms"<<endl;

/*
	CHECK(cudaMemcpy(Bt, d_temp1, Bt_size, cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(inner_inv, d_temp_inner_inv, innerinv_size, cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(Zden, d_Zden, Zden_size, cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(Z0, d_Z0, Z_size, cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(Z, d_Z, Z_size, cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(Q_re, d_Q_re, Q_size, cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(M, d_M, Q_size, cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(Y, d_Y, Q_size, cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(MminusZ,d_MminusZ,Q_size,cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(ZminusZO,d_ZminusZ0,Q_size,cudaMemcpyDeviceToHost));
*/

	//gpu memory free
	CHECK(cudaFree(d_xy));
	CHECK(cudaFree(d_B));
	CHECK(cudaFree(d_Bt));
	CHECK(cudaFree(d_bbt));
	CHECK(cudaFree(d_Zden));
	CHECK(cudaFree(d_Z));
	CHECK(cudaFree(d_Z0));
	CHECK(cudaFree(d_Zden_bbt));
	CHECK(cudaFree(d_T));
	CHECK(cudaFree(d_tempNum1));
	CHECK(cudaFree(d_tempNum2));
	CHECK(cudaFree(d_Znum));
	CHECK(cudaFree(d_M));
	CHECK(cudaFree(d_Y));
	CHECK(cudaFree(d_Q));
	CHECK(cudaFree(d_Q_re));
	CHECK(cudaFree(d_C));
	CHECK(cudaFree(d_MminusZ));	
	CHECK(cudaFree(d_ZminusZ0));
	CHECK(cudaFree(d_mui_B));
	CHECK(cudaFree(d_Bt_mui));
	CHECK(cudaFree(d_temp_inner));
	CHECK(cudaFree(d_temp_inner_inv));
	CHECK(cudaFree(d_temp1));

	CHECK(cudaEventDestroy(kernel_start));
	CHECK(cudaEventDestroy(kernel_stop));

}
