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

#define TILE_MN 6
#define TILE_MM 4
#define TILE_WIDTH 16

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

__global__ void transposeOnGPU(float *d_B, float *d_B_t, int ny, int nx)
{
	unsigned int ix = threadIdx.x + (blockIdx.x*blockDim.x);
	unsigned int iy = threadIdx.y + (blockIdx.y*blockDim.y);

	if (ix < nx && iy < ny)
	{
		d_B_t[ix*nx + iy] = d_B[iy*nx + ix];
	}
}


__global__ void MatVecMulcol(float *A, float *B, float *I_m, int m, int n)
{
	int col = threadIdx.x + blockIdx.x * blockDim.x; 	
	int row = threadIdx.y + blockIdx.y * blockDim.y; 
	
	if(col < n && row < m)
	{
		B[col*m+row] = A[col*m+row] * I_m[col];	
	}	
}

__global__ void MatVecMulrow(float *A, float *B, float *I_m, int m, int n)
{
	int col = threadIdx.x + blockIdx.x * blockDim.x; 	
	int row = threadIdx.y + blockIdx.y * blockDim.y; 
	
	if(col < n && row < m)
	{
		B[col*m+row] = A[col*m+row] * I_m[row];	
	}	
}


__global__ void MatmulplusI(float *A, float *B, float *C, int A_rows, int A_cols, int B_rows, int B_cols, int C_rows, int C_cols)
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

	for (int m = 0; m < (A_cols-1)/TILE_WIDTH+1; ++m) 
	{
		if (Row < A_rows && m*TILE_WIDTH+tx < A_cols)
		{
			ds_M[ty][tx] = A[Row*A_cols + m*TILE_WIDTH+tx];
		}
		else
		{
			ds_M[ty][tx] = 0;
		}

		if (Col < B_cols && m*TILE_WIDTH+ty < B_rows)
		{
			ds_N[ty][tx] = B[(m*TILE_WIDTH+ty)*B_cols+Col];
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

	if (Row < C_rows && Col < C_cols)
	{
		if(Row==Col)
		{
		C[Row*C_cols+Col] = fSum+1;
		}
		else
		{	
		C[Row*C_cols+Col] = fSum;
		}
	}
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

__global__ void IminusMatmul(float *A, float *B, float *C, float mu_inv, int A_rows, int A_cols, int B_rows, int B_cols, int C_rows, int C_cols)
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

	for (int m = 0; m < (A_cols-1)/TILE_WIDTH+1; ++m) 
	{
		if (Row < A_rows && m*TILE_WIDTH+tx < A_cols)
		{
			ds_M[ty][tx] = A[Row*A_cols + m*TILE_WIDTH+tx];
		}
		else
		{
			ds_M[ty][tx] = 0;
		}

		if (Col < B_cols && m*TILE_WIDTH+ty < B_rows)
		{
			ds_N[ty][tx] = B[(m*TILE_WIDTH+ty)*B_cols+Col];
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

	if (Row < C_rows && Col < C_cols)
	{
		if(Row==Col)
		{
		C[Row*C_cols+Col] = mu_inv - fSum;
		}
		else
		{	
		C[Row*C_cols+Col] = 0.0f - fSum;
		}
	}
}

__global__ void initializeGPU(float *d_B, float *d_B_t, int ny, int nx)
{
	unsigned int ix = threadIdx.x + (blockIdx.x*blockDim.x);
	unsigned int iy = threadIdx.y + (blockIdx.y*blockDim.y);

	if (ix < nx && iy < ny)
	{
		d_B_t[iy*nx + ix] = d_B[iy*nx + ix];
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
	__shared__ float row[TILE_WIDTH];
	__shared__ float rowI[TILE_WIDTH];

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


/*
__global__ void	Zden_caclGPU(float *B,float *B_transpose,float *Zden, float *temp_mui_B, float *temp_Bt_mui, float *temp_mult, float *temp_I, float *temp_inv,float mu_inv,int row1,int col);
{
	for(int i = 0;i<m;i++)
	{
		I_m[i] = mu_inv;
	}
	for(int i = 0;i<m;i++)
	{
		for(int j=0;j<n;j++)
		{
			temp_mui_B[(i*n)+j] = I_m[i] * B[(i*n)+j];
			temp_Bt_mui[(i*n)+j] = Bt[(i*n)+j] * I_m[j];
		}
	}

	
	float fSum;
        for (int i = 0; i < n; i++)
        {
                for (int j = 0; j < n; j++)
                {
                fSum = 0.0;
                        for (int k = 0; k < m; k++)
                        {
                        fSum += (Bt[(i*m) + k] * temp_mui_B[(k*n) + j]);
                        }
		if(i==j)
		{
                temp_inv[(i*n) + j] = fSum + 1;
		}
		else
		{
                temp_inv[(i*n) + j] = fSum;
		}
                }
        }	

	//matInv(temp_inv,n);
	eye(temp_I,n,n);
	cpuInverseOfMatrix(temp_inv, temp_I, n);
	
	for (int i = 0; i < n; i++)
        {
                for (int j = 0; j < m; j++)
                {
                fSum = 0.0;
                        for (int k = 0; k < n; k++)
                        {
                        fSum += (temp_I[(i*n) + k] * temp_Bt_mui[(k*m) + j]);
                        }
                temp_mult[(i*m) + j] = fSum;
                }
        }	

	for (int i = 0; i < m; i++)
        {
                for (int j = 0; j < m; j++)
                {
                fSum = 0.0;
                        for (int k = 0; k < n; k++)
                        {
                        fSum += (temp_mui_B[(i*n) + k] * temp_mult[(k*m) + j]);
                        }
		if(i==j)
		{
			Zden[(i*m)+j] = I_m[i] - fSum;
		}
		else
		{
			Zden[(i*m)+j] = 0.0 - fSum;
		}
                }
        }	
       

}

__host__ void eye(float *I, int m, int n)
{
        for(int i=0;i<n;i++)
        {
                for(int j=0;j<n;j++)
                {
                        if(i==j)
                        {
                                I[i*n+j] = 1.0;
                        }
                        else
                        {
                                I[i*n+j] = 0.0;
                        }
                }
        }
}


void initializeGPU(float *variable,float *variable2, int col, int row)
{
	for (int i = 0;i < row;i++)
	{
		for (int j = 0;j < col;j++)
		{
		variable[(i * col) + j] = variable2[(i * col) + j];
		}
	}
}

void calculateZ_preZden(float *Z,float *Zden,float *xy, float *E, float *T, float *B_transpose, float mu, float *M, float *Y,const int row,const int col,const int row1)
{
	//calculateZ_preZden(Z, Zden, xy, E, T, B_transpose,mu_orig,M,Y,row,col,row1);

        float *temp = new float [row*col];
        float *temp2 = new float [row*row1];
        float *temp3 = new float [row*row1];
        float *Znum = new float [row*row1];
        int status = 0;

        //numerator
        //temp = (W-E-T*ones(1,p))
        for (int i = 0;i < row;i++)
        {
                for (int j = 0;j < col;j++)
                {
                temp[(i*col) + j] = xy[(i*col) + j] - E[(i*col) + j] - T[i];
                }
        }
        //temp2 = temp * B'
        //GPU//
	//gpuMultShared(temp, B_transpose, temp2, row, col, col, row1);

        //Znum = ((W-E-T*ones(1,p))*B'+mu*M+Y) 
        sumOfMatrix(Znum,temp2, M, Y, mu, row, row1);

	//Z = ((W-E-T*ones(1,p))*B'+mu*M+Y)/(BBt+mu*eye(3*k))
	//GPU//
	gpuMultShared(Znum, Zden, Z, row, row1, row1, row1);

	delete [] temp;
        delete [] temp2;
        delete [] temp3;
        delete [] Znum;

}

void calculateQ(float *Q,float *Q_re, float *Z, float *Y,float mu, int row, int row1,int data_size)
{
	float oneovermu;
	oneovermu = 1/mu;
	for (int i = 0;i < row;i++)
        {
                for (int j = 0;j < row1;j++)
                {
                Q[(i*row1) + j] = Z[(i*row1) + j] - ((oneovermu)*Y[(i*row1) + j]) ;
                }
        }

	for(int i = 0;i < data_size;i++)
        {
                for(int j = 0;j<2;j++)
                {
                        for(int k=0;k<3;k++)
                        {

                        Q_re[(i*6)+(j * 3) + k] = Q[(3 * i) + (j*row1) + k];
                        }
                }
        }

}

void updateDualvariable(float *Y,float mu,float *M,float *Z,int row,int row1)
{
	for(int i=0;i<row;i++)
	{
		for(int j = 0;j<row1;j++)
		{
			Y[(i*row1)+j] += mu*(M[(i*row1)+j] - Z[(i*row1)+j]); 
		}
	}
}

void resCalc(float *PrimRes, float *DualRes, float *M, float *Z, float *ZO,float mu, int row, int row1)
{
	float *MminusZ = new float [row*row1];
	float *ZminusZO = new float [row*row1];

	for(int i = 0; i< row ;i++)
	{
		for(int j = 0; j<row1 ; j++)
		{
			MminusZ[(i*row1)+j] = M[(i*row1)+j] - Z[(i*row1)+j];
			ZminusZO[(i*row1)+j] = Z[(i*row1)+j] - ZO[(i*row1)+j];
		}
	}
	
		
	*PrimRes = febNorm(MminusZ,row,row1)/febNorm(ZO,row,row1);
	*DualRes = mu * febNorm(ZminusZO,row,row1)/febNorm(ZO,row,row1);
	
	delete[] MminusZ;
	delete[] ZminusZO;
}
*/

__host__ void loop_cu(float *xy, float *B, float *B_t, float *Z, float *ZO,float *Zden, float *Y, float *Q, float *Q_re,float *M, float *C,float *E, float *T, float *iden,float *I_m, float mu, float constant, int row, int col, int row1, int col1, int data_size,float *h_temp_Bt_mui)
{
	float *d_xy, *d_B, *d_B_t, *d_Z, *d_ZO, *d_Y, *d_Q, *d_Q_re, *d_M, *d_C, *d_E, *d_T, *d_Zden, *d_Im;

	//local arrays
	float *temp_mui_B, *temp_Bt_mui, *temp_mult, *temp_I, *temp_inv,*d_bbt;

	const int xy_size = row*col*sizeof(float);
	const int B_size = row1*col1*sizeof(float);
	const int Z_size = row*row1*sizeof(float);
	const int Y_size = row*row1*sizeof(float);
	const int Q_size = row*row1*sizeof(float);
	const int M_size = row*row1*sizeof(float);
	const int C_size = data_size*sizeof(float);
	const int E_size = row*col*sizeof(float);
	const int T_size = row*sizeof(float);
	const int inv_size = row1*row1*sizeof(float);
	const int winv_size = col*col*sizeof(float);
	const int Im_size = row1*sizeof(float);

	//memory allocation on GPU
	CHECK(cudaMalloc((void**)&d_xy,xy_size));
	CHECK(cudaMalloc((void**)&d_B,B_size));
	CHECK(cudaMalloc((void**)&d_B_t,B_size));
	CHECK(cudaMalloc((void**)&d_Z,Z_size));
	CHECK(cudaMalloc((void**)&d_ZO,Z_size));
	CHECK(cudaMalloc((void**)&d_Y,Y_size));
	CHECK(cudaMalloc((void**)&d_Q,Q_size));
	CHECK(cudaMalloc((void**)&d_Q_re,Q_size));
	CHECK(cudaMalloc((void**)&d_M,M_size));
	CHECK(cudaMalloc((void**)&d_C,C_size));
	CHECK(cudaMalloc((void**)&d_E,E_size));
	CHECK(cudaMalloc((void**)&d_T,T_size));
	CHECK(cudaMalloc((void**)&d_Zden,inv_size));
	
	CHECK(cudaMalloc((void**)&d_bbt,inv_size));
	CHECK(cudaMalloc((void**)&temp_mui_B,B_size));
	CHECK(cudaMalloc((void**)&temp_Bt_mui,B_size));
	CHECK(cudaMalloc((void**)&temp_mult,B_size));
	CHECK(cudaMalloc((void**)&temp_I,winv_size));
	CHECK(cudaMalloc((void**)&temp_inv,winv_size));
	CHECK(cudaMalloc((void**)&iden,winv_size));
	CHECK(cudaMalloc((void**)&d_Im,Im_size));

	//memory copy
	CHECK(cudaMemcpy(d_xy,xy,xy_size,cudaMemcpyHostToDevice));	
	CHECK(cudaMemcpy(d_B_t,B_t,B_size,cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_B,B,B_size,cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_Z,Z,Z_size,cudaMemcpyHostToDevice));	
	CHECK(cudaMemcpy(d_ZO,ZO,Z_size,cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_Y,Y,Y_size,cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_Q,Q,Q_size,cudaMemcpyHostToDevice));	
	CHECK(cudaMemcpy(d_Q_re,Q_re,Q_size,cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_M,M,M_size,cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_C,C,C_size,cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_E,E,E_size,cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_T,T,T_size,cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_Im,I_m,Im_size,cudaMemcpyHostToDevice));
	//CHECK(cudaMemcpy(temp_mui_B,h_temp_mui_B,Im_size,cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(temp_Bt_mui,h_temp_Bt_mui,Im_size,cudaMemcpyHostToDevice));

	int dimx_transpose = 16;
	int dimy_transpose = 16;
	dim3 block_transpose(dimx_transpose,dimy_transpose);
	dim3 grid_transpose((col+block_transpose.x-1)/block_transpose.x,(row1+block_transpose.y-1)/block_transpose.y);

	dim3 dimGrid_bbt((row1-1)/TILE_WIDTH+1, (row1-1)/TILE_WIDTH+1, 1);
	dim3 dimBlock_bbt(TILE_WIDTH, TILE_WIDTH, 1);
	
	int dimx_Zden = 16;
	int dimy_Zden = 16;
	dim3 block_Zden(dimx_Zden,dimy_Zden);
	dim3 grid_Zden((row1+block_Zden.x-1)/block_Zden.x,(col+block_Zden.y-1)/block_Zden.y);

	int dimx_Zcalc = 16;
	int dimy_Zcalc = 16;
	dim3 block_Zcalc(dimx_Zcalc,dimy_Zcalc);
	dim3 grid_Zcalc((col+block_Zcalc.x-1)/block_Zcalc.x,(row1+block_Zcalc.y-1)/block_Zcalc.y);

	int dimx_Zcalc1 = 16;
	int dimy_Zcalc1 = 16;
	dim3 block_Zcalc1(dimx_Zcalc1,dimy_Zcalc1);
	dim3 grid_Zcalc1((col+block_Zcalc1.x-1)/block_Zcalc1.x,(row1+block_Zcalc1.y-1)/block_Zcalc1.y);
	cout << "2D Grid Dimension" << endl;
	cout << "\tNumber of Blocks along X dimension: " << grid_Zcalc1.x << endl;
	cout << "\tNumber of Blocks along Y dimension: " << grid_Zcalc1.y << endl;
	cout << "2D Block Dimension" << endl;
	cout << "\tNumber of threads along X dimension: " << block_Zcalc1.x << endl;
	cout << "\tNumber of threads along Y dimension: " << block_Zcalc1.y << endl;


	dim3 block_inner(TILE_WIDTH, TILE_WIDTH, 1);
	dim3 grid_inner((col-1)/TILE_WIDTH+1, (col-1)/TILE_WIDTH+1, 1);
	
	dim3 block_outer(TILE_WIDTH, TILE_WIDTH, 1);
	dim3 grid_outer((row1-1)/TILE_WIDTH+1, (row1-1)/TILE_WIDTH+1, 1);
	
	int dimx_fixcol = col;
        int dimy_fixcol = 1;
        dim3 block_fixcol(dimx_fixcol,dimy_fixcol);  
        dim3 grid_fixcol(col,1); 

	int dimx2 = col;
        int dimy2 = 1;
        dim3 block2(dimx2,dimy2);     
        dim3 grid2(1,1); 

        int dimx3 = 32;
        int dimy3 = 32;
        dim3 block3(dimx3,dimy3); 
	dim3 grid3((col+block3.x-1)/block3.x,(col+block3.y-1)/block3.y);

	int dimx_ini = 16;
	int dimy_ini = 16;
	dim3 block_ini(dimx_ini,dimy_ini);
	dim3 grid_ini((row1+block_ini.x-1)/block_ini.x,(row+block_ini.y-1)/block_ini.y);


	int dimx_Qcalc = 16;
	int dimy_Qcalc = 16;
	dim3 block_Qcalc(dimx_Qcalc,dimy_Qcalc);
	dim3 grid_Qcalc((row1+block_Qcalc.x-1)/block_Qcalc.x,(col+block_Qcalc.y-1)/block_Qcalc.y);

	int dimx_proxnorm = 16;
	int dimy_proxnorm = 16;
	dim3 block_proxnorm(dimx_proxnorm,dimy_proxnorm);
	dim3 grid_proxnorm((row1+block_proxnorm.x-1)/block_proxnorm.x,(col+block_proxnorm.y-1)/block_proxnorm.y);

	int dimx_dual = 16;
	int dimy_dual = 16;
	dim3 block_dual(dimx_dual,dimy_dual);
	dim3 grid_dual((row1+block_dual.x-1)/block_dual.x,(col+block_dual.y-1)/block_dual.y);

	int dimx_res = 16;
	int dimy_res = 16;
	dim3 block_res(dimx_res,dimy_res);
	dim3 grid_res((row1+block_res.x-1)/block_res.x,(col+block_res.y-1)/block_res.y);


	transposeOnGPU << <grid_transpose, block_transpose >> >(d_B, d_B_t, row1, col);
	cudaThreadSynchronize();
	//Launch the GPU Kernel here
	matrixMultiply<<<dimGrid_bbt, dimBlock_bbt>>>(d_B, d_B_t, d_bbt, row1, col, col, row1, row1, row1);


/*
	MatVecMulcol<< <grid_Zcalc,block_Zcalc >> >(d_B,temp_mui_B , d_Im, row1, col);
	cudaThreadSynchronize();
	MatVecMulrow<< <grid_Zcalc,block_Zcalc >> >(d_B_t,temp_Bt_mui , d_Im, row1, col);
	
	cudaThreadSynchronize();
	
	MatmulplusI<< <grid_inner,block_inner >> >(d_B_t, temp_mui_B, temp_inv, col1, row1, row1, col1, col1, col1);

	cudaThreadSynchronize();
	
	check_diag_zero << <grid3, block3 >> >(temp_inv, iden, col);
	for (int i = 0; i<col; i++)
	{
		fixRow_shared << <grid2, block2 >> >(temp_inv, iden, col, i);
		fixColumn_shared << <grid_fixcol, block_fixcol >> >(temp_inv, iden, col, i);
	}

	cudaThreadSynchronize();

	matrixMultiply<<<grid_inner, block_inner>>>(iden, temp_Bt_mui, temp_mult, col, col, col, row1, col, row1);
	
	cudaThreadSynchronize();

	MatmulplusI<< <grid_outer,block_outer >> >(temp_mui_B, temp_mult, d_Zden, row1, col1, col1, row1, row1, row1);
*/


/*
	for(int iter=0; iter < 1;iter++)
	{
		initializeGPU<< <grid_ini,block_ini >> >(d_ZO,d_Z,row1,row);
		if(flag == 1)
		{
			Zden_caclGPU(B,B_transpose,Zden,mu,row1,col);
		}
	
		calculateZ_preZden(Z, Zden,xy, E, T, B_transpose,mu,M,Y,row,col,row1);
		calculateQ(Q,Q_re,Z,Y,mu,row,row1,data_size);
		
		gpuProx_2norm(Q_re,M,C,lam/mu,row,row1,data_size);	

		updateDualvariable(Y,mu,M,Z,row,row1);
		
		resCalc(&PrimRes, &DualRes, M, Z, ZO, mu, row, row1);
		
		if((PrimRes < tol) && (DualRes < tol))
		{
		break;
		}
		else
		{
			if(PrimRes > (10*DualRes))
			{
				mu = 2 * mu;
				flag = 1;
			}
			else if(DualRes > (10*PrimRes))
			{
				mu = mu/2;
				flag = 1;
			}
			else
			{
				flag = 0;
			}
		}

	}
*/
	CHECK(cudaMemcpy(Zden, d_bbt, inv_size, cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(B, d_B, B_size, cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(B_t, d_B_t, B_size, cudaMemcpyDeviceToHost));


	//gpu memory free
	CHECK(cudaFree(d_xy));
	CHECK(cudaFree(d_B));
	CHECK(cudaFree(d_B_t));
	CHECK(cudaFree(d_Z));
	CHECK(cudaFree(d_ZO));
	CHECK(cudaFree(d_Y));
	CHECK(cudaFree(d_Q));
	CHECK(cudaFree(d_Q_re));
	CHECK(cudaFree(d_M));
	CHECK(cudaFree(d_C));
	CHECK(cudaFree(d_E));
	CHECK(cudaFree(d_T));
	CHECK(cudaFree(d_Zden));
	
	CHECK(cudaFree(temp_mui_B));
        CHECK(cudaFree(temp_Bt_mui));
        CHECK(cudaFree(temp_mult));
        CHECK(cudaFree(temp_I));
        CHECK(cudaFree(temp_inv));
	

}
