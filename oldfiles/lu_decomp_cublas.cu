#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define cudacall(call)                                                                                                          \
    do                                                                                                                          \
    {                                                                                                                           \
        cudaError_t err = (call);                                                                                               \
        if(cudaSuccess != err)                                                                                                  \
        {                                                                                                                       \
            fprintf(stderr,"CUDA Error:\nFile = %s\nLine = %d\nReason = %s\n", __FILE__, __LINE__, cudaGetErrorString(err));    \
            cudaDeviceReset();                                                                                                  \
            exit(EXIT_FAILURE);                                                                                                 \
        }                                                                                                                       \
    }                                                                                                                           \
    while (0)

#define cublascall(call)                                                                                        \
    do                                                                                                          \
    {                                                                                                           \
        cublasStatus_t status = (call);                                                                         \
        if(CUBLAS_STATUS_SUCCESS != status)                                                                     \
        {                                                                                                       \
            fprintf(stderr,"CUBLAS Error:\nFile = %s\nLine = %d\nCode = %d\n", __FILE__, __LINE__, status);     \
            cudaDeviceReset();                                                                                  \
            exit(EXIT_FAILURE);                                                                                 \
        }                                                                                                       \
                                                                                                                \
    }                                                                                                           \
    while(0)


void invert_device(double* src_d, double* dst_d, int n)
{
    cublasHandle_t handle;
    cublascall(cublasCreate_v2(&handle));

    int batchSize = 1;

    int *P, *INFO;

    cudacall(cudaMalloc((void**)&P,n * batchSize * sizeof(int)));
    cudacall(cudaMalloc((void**)&INFO,batchSize * sizeof(int)));

    int lda = n;

    double *A[] = { src_d };
    double** A_d;
    cudacall(cudaMalloc((const double**)&A_d,sizeof(A)));
    cudacall(cudaMemcpy(A_d,A,sizeof(A),cudaMemcpyHostToDevice));

    cublascall(cublasDgetrfBatched(handle,n,A_d,lda,P,INFO,batchSize));

    int INFOh = 0;
    cudacall(cudaMemcpy(&INFOh,INFO,sizeof(int),cudaMemcpyDeviceToHost));

    if(INFOh == n)
    {
        fprintf(stderr, "Factorization Failed: Matrix is singular\n");
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }

    double* C[] = { dst_d };
    double** C_d;
    cudacall(cudaMalloc((const double**)&C_d,sizeof(C)));
    cudacall(cudaMemcpy(C_d,C,sizeof(C),cudaMemcpyHostToDevice));

    cublascall(cublasDgetriBatched(handle,n,(const double**)A_d,lda,P,C_d,lda,INFO,batchSize));

    cudacall(cudaMemcpy(&INFOh,INFO,sizeof(int),cudaMemcpyDeviceToHost));

    if(INFOh != 0)
    {
        fprintf(stderr, "Inversion Failed: Matrix is singular\n");
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }

    cudaFree(P), cudaFree(INFO), cublasDestroy_v2(handle);
}

void invert(double* src, double* dst, int n)
{
    double* src_d, *dst_d;

    cudacall(cudaMalloc((void**)&src_d,n * n * sizeof(double)));
    cudacall(cudaMemcpy(src_d,src,n * n * sizeof(double),cudaMemcpyHostToDevice));
    cudacall(cudaMalloc((void**)&dst_d,n * n * sizeof(double)));

    invert_device(src_d,dst_d,n);

    cudacall(cudaMemcpy(dst,dst_d,n * n * sizeof(double),cudaMemcpyDeviceToHost));

    cudaFree(src_d);
    cudaFree(dst_d);
}

void test_invert()
{
    const int n = 3;

/*    //Random matrix with full pivots
    double full_pivots[n*n] = { 0.5, 3, 4, 
                                1, 3, 10, 
                                4 , 9, 16 };
*/
    //Almost same as above matrix with first pivot zero

    double a[n*n] = { 0, 3, 4, 
          1, 3, 10,
          4 , 9, 16 };

/*    double zero_pivot_col_major[n*n] = { 0, 1, 4, 
                                        3, 3, 9,
                                        4 , 10, 16 };

    double another_zero_pivot[n*n] = { 0, 3, 4, 
                                      1, 5, 6,
                                      9, 8, 2 };

    double another_full_pivot[n * n] = { 22, 3, 4, 
                                        1, 5, 6,
                                        9, 8, 2 };

    double singular[n*n] = {1,2,3,
                           4,5,6,
                           7,8,9};
*/
/*double *a = new double [n*n];
for(int i=0;i<n;i++)
{
for(int j=0;j<n;j++)
{
a[(i*n)+j] = ((double) rand() / 2) + 1;
}
}
*/

    fprintf(stdout, "Input:\n\n");
    for(int i=0; i<n; i++)
    {
        for(int j=0; j<n; j++)
            fprintf(stdout,"%f\t",a[i*n+j]);
        fprintf(stdout,"\n");
    }

    fprintf(stdout,"\n\n");

    invert(a,a,n);

    fprintf(stdout, "Inverse:\n\n");
    for(int i=0; i<n; i++)
    {
        for(int j=0; j<n; j++)
            fprintf(stdout,"%f\t",a[i*n+j]);
        fprintf(stdout,"\n");
    }

}

int main()
{
    test_invert();

    return 0;
}
