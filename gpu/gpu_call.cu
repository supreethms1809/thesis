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

#define TILE_DIM 32;
#define BLOCK_ROWS 8;
#define TILE_MN 6
#define TILE_MM 4
#define TILE_WIDTH 16
#define TILE_WIDTH_INVERSE 1536

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


__global__ void transposeOnGPU(float *d_B, float *d_Bt, int rows, int cols)
{
	unsigned int iy = threadIdx.x + (blockIdx.x*blockDim.x);
	unsigned int ix = threadIdx.y + (blockIdx.y*blockDim.y);

	if (ix < rows && iy < cols)
	{
		d_Bt[iy*rows + ix] = d_B[ix*cols + iy];
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

__global__ void sumOfMatrixGPU(float *d_Znum,float *d_tempNum2,float *d_M,float *d_Y,float mu,int row,int col)
{
	unsigned int iy = threadIdx.x + (blockIdx.x*blockDim.x);
	unsigned int ix = threadIdx.y + (blockIdx.y*blockDim.y);

	if (ix < row && iy < col)
	{
		d_Znum[ix*col + iy] = d_tempNum2[ix*col + iy]+mu*d_M[ix*col + iy]+d_Y[ix*col + iy];
	}
}

__host__ void loop(float *xy,float *B,float *Bt,float *Zden,float *Z,float *Z0,float *T,float *M,float *Y,int row,int col,int row1,int col1,float mu, float *bbt)
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
	

	CHECK(cudaMemcpy(d_xy,xy,xy_size,cudaMemcpyHostToDevice));	
	CHECK(cudaMemcpy(d_B,B,B_size,cudaMemcpyHostToDevice));	
	CHECK(cudaMemcpy(d_Bt,Bt,Bt_size,cudaMemcpyHostToDevice));	
	CHECK(cudaMemcpy(d_Zden,Zden,Zden_size,cudaMemcpyHostToDevice));	
	CHECK(cudaMemcpy(d_Z,Z,Z_size,cudaMemcpyHostToDevice));	
	CHECK(cudaMemcpy(d_T,T,T_size,cudaMemcpyHostToDevice));	
	

	int dimx_transpose = 16;
	int dimy_transpose = 16;
	dim3 block_transpose(dimx_transpose,dimy_transpose);
	dim3 grid_transpose((col1+block_transpose.x-1)/block_transpose.x,(row1+block_transpose.y-1)/block_transpose.y);

	int dimx_bbt = 16;
	int dimy_bbt = 16;
	dim3 block_bbt(dimx_bbt,dimy_bbt);
	dim3 grid_bbt((row1+block_bbt.x-1)/block_bbt.x,(row1+block_bbt.y-1)/block_bbt.y);

	//2D grid and 2D block
        int dimx2 = row1;
        int dimy2 = 1;
        dim3 block2(dimx2,dimy2);   
        dim3 grid2(1,1); 
	
	int dimx_fixcol = row1;
        int dimy_fixcol = 1;
        dim3 block_fixcol(dimx_fixcol,dimy_fixcol);     
	dim3 grid_fixcol(row1,1); 

        int dimx3 = 32;
        int dimy3 = 32;
        dim3 block3(dimx3,dimy3); 
	dim3 grid3((row1+block3.x-1)/block3.x,(row1+block3.y-1)/block3.y);

	int dimx_ini = 16;
	int dimy_ini = 16;
	dim3 block_ini(dimx_ini,dimy_ini);
	dim3 grid_ini((row1+block_ini.x-1)/block_ini.x,(row+block_ini.y-1)/block_ini.y);

	int dimx_tempnum1 = 16;
	int dimy_tempnum1 = 16;
	dim3 block_tempnum1(dimx_tempnum1,dimy_tempnum1);
	dim3 grid_tempnum1((col+block_ini.x-1)/block_ini.x,(row+block_ini.y-1)/block_ini.y);



	cout << "2D Grid Dimension" << endl;
	cout << "\tNumber of Blocks along X dimension: " << grid_tempnum1.x << endl;
	cout << "\tNumber of Blocks along Y dimension: " << grid_tempnum1.y << endl;
	cout << "2D Block Dimension" << endl;
	cout << "\tNumber of threads along X dimension: " << block_tempnum1.x << endl;
	cout << "\tNumber of threads along Y dimension: " << block_tempnum1.y << endl;

	transposeOnGPU << <grid_transpose, block_transpose >> >(d_B, d_Bt, row1, col1);
	cudaThreadSynchronize();

	matrixMultiply<<<grid_bbt, block_bbt>>>(d_B, d_Bt, d_bbt, row1, col, col, row1, row1, row1);
	cudaThreadSynchronize();
	
	addmu_diagonal<<<grid_bbt,block_bbt>>>(d_bbt,d_Zden_bbt,mu,row1,row1);
	
	check_diag_zero << <grid3, block3 >> >(d_Zden_bbt, d_Zden, row1);
	for (int i = 0; i<row1; i++)
	{
		fixRow_shared << <grid2, block2 >> >(d_Zden_bbt, d_Zden, row1, i);
		fixColumn_shared << <grid_fixcol, block_fixcol >> >(d_Zden_bbt, d_Zden, row1, i);
	}
	cudaThreadSynchronize();
	
	for(int iter=0;iter<1;iter++)
	{
		initializeZGPU<< <grid_ini,block_ini >> >(d_Z,d_Z0,row,row1);

		//Z calculation	
		//calculateZ_preZden(Z, Zden_inv,xy, E, T, B_transpose,mu,M,Y,row,col,row1);
		tempnum1Calc<<<grid_tempnum1,block_tempnum1>>>(d_tempNum1,d_xy,d_T,row,col);
		cudaThreadSynchronize();
		matrixMultiply<<<grid_ini,block_ini>>>(d_tempNum1,d_Bt,d_tempNum2,row,col,col1,row1,row,row1);
		cudaThreadSynchronize();
		sumOfMatrixGPU<<<grid_ini,block_ini>>>(d_Znum,d_tempNum2, d_M, d_Y, mu, row, row1);
		cudaThreadSynchronize();
		matrixMultiply<<<grid_ini,block_ini>>>(d_Znum,d_Zden,d_Z,row,row1,row1,row1,row,row1);
		cudaThreadSynchronize();

		

	}


	CHECK(cudaMemcpy(Bt, d_Bt, Bt_size, cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(bbt, d_bbt, bbt_size, cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(Zden, d_Zden, Zden_size, cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(Z0, d_Z0, Z_size, cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(Z, d_Z, Z_size, cudaMemcpyDeviceToHost));
	//CHECK(cudaMemcpy(Z, d_Znum, Z_size, cudaMemcpyDeviceToHost));

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

}
