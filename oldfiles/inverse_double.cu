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
int* cudaSgetrf(unsigned int n, double *dA);
void cudaSgetri(unsigned int n, double *dA, int *pivots);
void cudaStrtri(unsigned int n, double *dA);



/*
** cudaInvertMatrix
** Inverts a square matrix in place
** n - matrix dimension
** A - pointer to array of doubles representing the matrix in column-major order
*/

extern "C" void cudaInvertMatrix(unsigned int n, double *A)
{
	
int *pivots;
double *dA;

cublasInit();

cublasAlloc(n * n, sizeof(double), (void**)&dA);
cublasSetMatrix(n, n, sizeof(double), A, n, dA, n);

// Perform LU factorization
pivots = cudaSgetrf(n, dA);

// Perform inversion on factorized matrix
cudaSgetri(n, dA, pivots);

cublasGetMatrix(n, n, sizeof(double), dA, n, A, n);

cublasFree(dA);
cublasShutdown();

}



/*
** cudaSgetrf
** Performs an in-place LU factorization on a square matrix
** Uses the unblocked BLAS2 approach
*/

int *cudaSgetrf(unsigned int n, double *dA)
{

int i, pivot, *pivots;
double *offset, factor;

pivots = (int *) calloc(n, sizeof(int));

	for(i = 0; i < n; ++i)
	{
		pivots[i] = i;
	}
	for(i = 0; i < n - 1; i++) 
	{
		offset = dA + i*n + i;
		pivot = i - 1 + cublasIdamax(n - i, offset, 1);

			if(pivot != i) 
			{
				pivots[i] = pivot;
				cublasDswap(n, dA + pivot, n, dA + i, n);
			}
		cublasGetVector(1, sizeof(double), offset, 1, &factor, 1);
		cublasDscal(n - i - 1, 1 / factor, offset + 1, 1);
		cublasDger(n - i - 1, n - i - 1, -1.0f, offset + 1, 1, offset + n, n, offset + n + 1, n);
	}

return pivots;
}



/*
** cudaSgetri
** Computes the inverse of an LU-factorized square matrix
*/

void cudaSgetri(unsigned int n, double *dA, int *pivots)
{

int i;
double *dWork, *offset;

// Perform inv(U)
cudaStrtri(n, dA);

// Solve inv(A)*L = inv(U)
cublasAlloc(n - 1, sizeof(double), (void**)&dWork);

	for(i = n - 1; i > 0; --i) 
	{

		offset = dA + (i - 1)*n + i;
		cudaMemcpy(dWork, offset, (n - 1) * sizeof(double), cudaMemcpyDeviceToDevice);
		cublasDscal(n - i, 0, offset, 1);
		cublasDgemv('n', n, n - i, -1.0f, dA + i*n, n, dWork, 1, 1.0f, dA + (i-1)*n, 1);

	}

cublasFree(dWork);

// Pivot back to original order
	for(i = n - 1; i >= 0; --i)
	{
		if(i != pivots[i])
		{
			cublasDswap(n, dA + i*n, 1, dA + pivots[i]*n, 1);
		}
	}
}



/*
** cudaStrtri
** Computes the inverse of an upper triangular matrix in place
** Uses the unblocked BLAS2 approach
*/
void cudaStrtri(unsigned int n, double *dA)
{

int i;
double factor, *offset;

	for(i = 0; i < n; ++i) 
	{

		offset = dA + i*n;
		cublasGetVector(1, sizeof(double), offset + i, 1, &factor, 1);
		factor = 1 / factor;
		cublasSetVector(1, sizeof(double), &factor, 1, offset + i, 1);
		cublasDtrmv('u', 'n', 'n', i, dA, n, offset, 1);
		cublasDscal(i, -1 * factor, offset, 1);

	}

}
