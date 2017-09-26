#include <stdio.h>
#include "cblas.h"
#include "mkl_lapacke.h"

// inplace inverse n x n matrix A.
// matrix A is Column Major (i.e. firts line, second line ... *not* C[][] order)
// returns:
//   ret = 0 on success
//   ret < 0 illegal argument value
//   ret > 0 singular matrix

lapack_int matInv(double *A, unsigned n)
{
    int ipiv[n+1];
    lapack_int ret;

    ret =  LAPACKE_dgetrf(LAPACK_COL_MAJOR,
                          n,
                          n,
                          A,
                          n,
                          ipiv);

    if (ret !=0)
        return ret;


    ret = LAPACKE_dgetri(LAPACK_COL_MAJOR,
                       n,
                       A,
                       n,
                       ipiv);
    return ret;
}

int main()
{
    double A[] = {
        0.378589,   0.971711,   0.016087,   0.037668,   0.312398,
        0.756377,   0.345708,   0.922947,   0.846671,   0.856103,
        0.732510,   0.108942,   0.476969,   0.398254,   0.507045,
        0.162608,   0.227770,   0.533074,   0.807075,   0.180335,
        0.517006,   0.315992,   0.914848,   0.460825,   0.731980
    };

    for (int i=0; i<25; i++) {
        if ((i%5) == 0) putchar('\n');
        printf("%+12.8f ",A[i]);
    }
    putchar('\n');

    matInv(A,5);

    for (int i=0; i<25; i++) {
        if ((i%5) == 0) putchar('\n');
        printf("%+12.8f ",A[i]);
    }
    putchar('\n');
}
