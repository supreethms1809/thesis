#include <stdio.h>
#include <math.h>
#include <iostream>
#include <chrono>
#include <ctime>
#include "mkl_cblas.h"
#include "mkl_lapacke.h"
#include "magma.h"
#include "magma_v2.h"

using namespace std;
using namespace std::chrono;
// inplace inverse n x n matrix A.
// matrix A is Column Major (i.e. firts line, second line ... *not* C[][] order)
// returns:
//   ret = 0 on success
//   ret < 0 illegal argument value
//   ret > 0 singular matrix

extern "C" void cudaInvertMatrix(unsigned int n, double *A);

void ludcmp_sup(double *A,int n)
{
	int i,j,k;

	
	for(i=0;i<n;i++)
	{
		double sum = 0.0;
		for(j = 0; j < i;j++)
		{
			cout << "second : i = "<< i << "\t" << "j = "<<j<<endl;
			sum = A[(i*n)+j];
		
			for(k = 0;k<(j-1);k++)
			{
				cout << "coming inside second k : " << k <<" and i = " << i <<endl;
				sum -= A[(i*n)+k] * A[(k*n)+j] ;
			}	
		
			A[(i*n)+j] = sum;///A[(j*n)+n];
		}

		for(j=i;j<n;j++)
		{
			cout << "first : i = "<< i << "\t" << "j = "<<j<<endl;
			sum = A[(i*n)+j];
			cout << "sum before "<< sum <<endl;	
			for(k = 0;k<i;k++)
			{
				cout << "coming inside first k : " <<k<<endl;
				sum -= A[(i*n)+k] * A[(k*n)+j];
			}	
			cout << "sum after "<< sum <<endl;
			A[(i*n)+j] = sum;
		}
		cout << endl<<endl;
		for (int ii=0; ii<9; ii++) 
		{
        	if ((ii%3) == 0) putchar('\n');
        	printf("%+12.8f ",A[ii]);
		}
		cout << endl<<endl;

		//if(i > 0)
		//{
		//}

			cout << "next iteration "<< endl;		

	}

}

void ludcmp(double *a,int n)
{
int i,imax,j,k;
double big,dum,sum,temp;
float *vv = new float[n];                     

	//for(i=0;i<n;i++)
	//{
	//	vv[i] = 1;
	//	cout << "memory allocated "<<endl;
	//}

	for (i=0;i<n;i++) 
	{
		big=0.0;
		for (j=0;j<n;j++)
		{
			if ((temp=fabs(a[(i*n)+j])) > big) 
			{
			big=temp;
			}
		}
		vv[i]=1.0/big;
	}
	for (i=0;i<n;i++) 
	{
		for (j=0;j<i;j++) 
		{
			sum=a[(i*n)+j];
				for (k=0;k<(j-1);k++)
				{
				 sum -= a[(i*n)+k]*a[(k*n)+j];
				}
			a[(i*n)+j]=sum;
		}
		big=0.0;

		for (j=i;j<n;j++) 
		{
		sum=a[(i*n)+j];
			for (k=0;k<i;k++)
			{
			sum -= a[(i*n)+k]*a[(k*n)+j];
			}
		a[(i*n)+j]=sum;
			if ( (dum=vv[j]*fabs(sum)) >= big)
			{
			big=dum;
			imax=j;
			}
		}

	
		if (i != imax) 
		{
			for (k=0;k<n;k++) 
			{
			dum=a[(imax*n)+k];
			a[(imax*n)+k]=a[(i*n)+k];
			a[(i*n)+k]=dum;
			}
			vv[imax]=vv[i];
		}
		//if (a[(j*n)+j] == 0.0) a[(j*n)+j]=TINY;
		if (i != n) 
		{
			dum=1.0/(a[(i*n)+i]);
			for (j=i;j<n;j++) 
			{
			a[(i*n)+j] *= dum;
			}
		}
	}	
delete[] vv;
}

magma_int_t matInv_magma(double *A, magma_int_t n)
{
	magma_int_t ipiv[n];
        magma_int_t ret,lwork,ldwork;
        
        ldwork = n * magma_get_dgetri_nb(n);
        double *dwork = new double [ldwork];

        
        magma_dgetrf(n,n,A,n,ipiv,&ret);
 cout <<endl<< "Matrix B inside magma"<<endl;
for (int i=0; i<9; i++) {
        if ((i%3) == 0) putchar('\n');
        printf("%+12.8f ",A[i]);
    }
    putchar('\n');                
        
        magma_dgetri_gpu( n, A, n, ipiv, dwork, ldwork, &ret );
 cout <<endl<< "Matrix B inside magma after dgetri"<<endl;
for (int i=0; i<9; i++) {
        if ((i%3) == 0) putchar('\n');
        printf("%+12.8f ",A[i]);
    }
    putchar('\n');
       return ret;

}


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

cout << endl<<"Matrix A inside lapack _ after dgetrf"<<endl;
for (int i=0; i<9; i++) {
        if ((i%3) == 0) putchar('\n');
        printf("%+12.8f ",A[i]);
    }
    putchar('\n');


    ret = LAPACKE_dgetri(LAPACK_COL_MAJOR,
                       n,
                       A,
                       n,
                       ipiv);
    return ret;
}


int main()
{
/*
    double A[] = {
        0.378589,   0.971711,   0.016087,   0.037668,   0.312398,
        0.756377,   0.345708,   0.922947,   0.846671,   0.856103,
        0.732510,   0.108942,   0.476969,   0.398254,   0.507045,
        0.162608,   0.227770,   0.533074,   0.807075,   0.180335,
        0.517006,   0.315992,   0.914848,   0.460825,   0.731980
    };
    double B[] = {
        0.378589,   0.971711,   0.016087,   0.037668,   0.312398,
        0.756377,   0.345708,   0.922947,   0.846671,   0.856103,
        0.732510,   0.108942,   0.476969,   0.398254,   0.507045,
        0.162608,   0.227770,   0.533074,   0.807075,   0.180335,
        0.517006,   0.315992,   0.914848,   0.460825,   0.731980
    };
*/
magma_init();
   double A[] = {
        1,   2,   3, 
        4,   5,   6, 
        7,   8,   10
    };
    double B[] = {
        1,   2,   3, 
        4,   5,   6, 
        7,   8,   10
    };
	
	/*high_resolution_clock::time_point t3,t4;	
	const int n =384;
	double *B = new double [n*n];
	for(int i =0;i<n;i++)
	{
		for(int j = 0;j<n;j++)
		{
			B[(i*n)+j] = i*j/fabs(i-j);
		}
	}
*/

    //for (int i=0; i<9; i++) {
    //   if ((i%3) == 0) putchar('\n');
    //    printf("%+12.8f ",A[i]);
    //}
    //putchar('\n');

	matInv(A,3);
	matInv_magma(B,3);
	//t3 = high_resolution_clock::now();
	//cudaInvertMatrix(n,B);
        //t4 = high_resolution_clock::now(); 
        //duration<double> time_span = duration_cast<duration<double>>(t4 - t3);    
        //cout << "Time in miliseconds for first section is : " << time_span.count() * 1000 << " ms" << endl;

//	delete[] B;
cout <<endl<< "matrix inverse of B"<<endl;
	for (int i=0; i<9; i++) {
        if ((i%3) == 0) putchar('\n');
        printf("%+12.8f ",B[i]);
    }
    putchar('\n');

cout <<endl<< "matrix inverse of A "<<endl;
    for (int i=0; i<9; i++) {
        if ((i%3) == 0) putchar('\n');
        printf("%+12.8f ",A[i]);
    }
    putchar('\n');
magma_finalize();
}
