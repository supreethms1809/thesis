#include <stdio.h>
#include <math.h>
#include <iostream>
#include "mkl_cblas.h"
#include "mkl_lapacke.h"

using namespace std;
// inplace inverse n x n matrix A.
// matrix A is Column Major (i.e. firts line, second line ... *not* C[][] order)
// returns:
//   ret = 0 on success
//   ret < 0 illegal argument value
//   ret > 0 singular matrix


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


    for (int i=0; i<9; i++) {
        if ((i%3) == 0) putchar('\n');
        printf("%+12.8f ",A[i]);
    }
    putchar('\n');

    matInv(A,3);
    ludcmp(B,3);

	for (int i=0; i<9; i++) {
        if ((i%3) == 0) putchar('\n');
        printf("%+12.8f ",B[i]);
    }
    putchar('\n');

    
    for (int i=0; i<9; i++) {
        if ((i%3) == 0) putchar('\n');
        printf("%+12.8f ",A[i]);
    }
    putchar('\n');
}
