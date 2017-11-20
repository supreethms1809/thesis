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

#define TILE_WIDTH 384

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


__host__ void gpuProx_2norm(float *Q, float *M, float *C, float constant, int row, int col, int data_size)
{
	float *d_Q,*d_M,*d_C;
	const int MatSizeInBytes = row*col*sizeof(float);
	const int CsizeInBytes = col*sizeof(float);

	//memory allocation on GPU
	CHECK(cudaMalloc((void**)&d_Q,MatSizeInBytes));
	CHECK(cudaMalloc((void**)&d_M,u_size));
	CHECK(cudaMalloc((void**)&d_C,sig_size));
	
	//data copy into GPU memory
	CHECK(cudaMemcpy(d_Q,Q,MatSizeInBytes,cudaMemcpyHostToDevice));

	//2D grid and 2D block
	int dimx = 1;
	int dimy = 1;
	dim3 block(dimx,dimy);
	dim3 grid((col+block.x-1)/block.x,1);
	cout << "threads in a block "<<block<<endl;
	cout << "blocks in a grid "<<grid <<endl;

	//copy back data from GPU
	CHECK(cudaMemcpy(M,d_M,MatSizeInBytes,cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(C,d_C,MatSizeInBytes,cudaMemcpyDeviceToHost));
	CHECK(cudaFree(d_Q));
	CHECK(cudaFree(d_M));
	CHECK(cudaFree(d_C));


}
