/*
** Simple matrix inversion routine using CUDA
**
** This reimplements the LAPACK sgetri and associated required routines
** by replacing the BLAS calls with CUBLAS calls.
**
** Byron Galbraith
** Department of Mathematics, Statistics, and Computer Science
** Marquette University
** 2009-04-30
*/

#include <cublas.h>

// Prototypes
int* cudaSgetrf(unsigned int n, float *dA);
void cudaSgetri(unsigned int n, float *dA, int *pivots);
void cudaStrtri(unsigned int n, float *dA);



/*
** cudaInvertMatrix
** Inverts a square matrix in place
** n - matrix dimension
** A - pointer to array of floats representing the matrix in column-major order
*/

extern "C" void cudaInvertMatrix(unsigned int n, float *A)

{
	
int *pivots;
float *dA;

cublasInit();

cublasAlloc(n * n, sizeof(float), (void**)&dA);
cublasSetMatrix(n, n, sizeof(float), A, n, dA, n);

// Perform LU factorization
pivots = cudaSgetrf(n, dA);

// Perform inversion on factorized matrix
cudaSgetri(n, dA, pivots);

cublasGetMatrix(n, n, sizeof(float), dA, n, A, n);

cublasFree(dA);
cublasShutdown();

}



/*
** cudaSgetrf
** Performs an in-place LU factorization on a square matrix
** Uses the unblocked BLAS2 approach
*/

int *cudaSgetrf(unsigned int n, float *dA)
{

int i, pivot, *pivots;
float *offset, factor;

pivots = (int *) calloc(n, sizeof(int));

	for(i = 0; i < n; ++i)
	{
		pivots[i] = i;
	}
	for(i = 0; i < n - 1; i++) 
	{
		offset = dA + i*n + i;
		pivot = i - 1 + cublasIsamax(n - i, offset, 1);

			if(pivot != i) 
			{
				pivots[i] = pivot;
				cublasSswap(n, dA + pivot, n, dA + i, n);
			}
		cublasGetVector(1, sizeof(float), offset, 1, &factor, 1);
		cublasSscal(n - i - 1, 1 / factor, offset + 1, 1);
		cublasSger(n - i - 1, n - i - 1, -1.0f, offset + 1, 1, offset + n, n, offset + n + 1, n);
	}

return pivots;
}



/*
** cudaSgetri
** Computes the inverse of an LU-factorized square matrix
*/

void cudaSgetri(unsigned int n, float *dA, int *pivots)
{

int i;
float *dWork, *offset;

// Perform inv(U)
cudaStrtri(n, dA);

// Solve inv(A)*L = inv(U)
cublasAlloc(n - 1, sizeof(float), (void**)&dWork);

	for(i = n - 1; i > 0; --i) 
	{

		offset = dA + (i - 1)*n + i;
		cudaMemcpy(dWork, offset, (n - 1) * sizeof(float), cudaMemcpyDeviceToDevice);
		cublasSscal(n - i, 0, offset, 1);
		cublasSgemv('n', n, n - i, -1.0f, dA + i*n, n, dWork, 1, 1.0f, dA + (i-1)*n, 1);

	}

cublasFree(dWork);

// Pivot back to original order
	for(i = n - 1; i >= 0; --i)
	{
		if(i != pivots[i])
		{
			cublasSswap(n, dA + i*n, 1, dA + pivots[i]*n, 1);
		}
	}
}



/*
** cudaStrtri
** Computes the inverse of an upper triangular matrix in place
** Uses the unblocked BLAS2 approach
*/
void cudaStrtri(unsigned int n, float *dA)
{

int i;
float factor, *offset;

	for(i = 0; i < n; ++i) 
	{

		offset = dA + i*n;
		cublasGetVector(1, sizeof(float), offset + i, 1, &factor, 1);
		factor = 1 / factor;
		cublasSetVector(1, sizeof(float), &factor, 1, offset + i, 1);
		cublasStrmv('u', 'n', 'n', i, dA, n, offset, 1);
		cublasSscal(i, -1 * factor, offset, 1);

	}

}