#include <stdio.h>
#include <string>
#include <iostream>
#include <fstream>
#include <math.h>
#include <stdlib.h>
#include <mkl.h>
#include <mkl_lapack.h>
#include <limits>
#include <chrono>
#include <ctime>
#include <omp.h>

using namespace std;

void print_matrix( char *desc, MKL_INT m, MKL_INT n, float *a) 
{
        MKL_INT i, j;
        printf( "\n %s\n", desc );
        for( i = 0; i < m; i++ ) 
	{
       		for( j = 0; j < n; j++ )
		{
		cout << a[i*n+j] << "\t";
		}
        cout << "\n" ;
	}
}
void cpuMatrixMult(float *A, float *B, float *C, int row, int col,int col2)
{
	float fSum;
	int count = 0;
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col2; j++)
		{
		fSum = 0.0;
			for (int k = 0; k < col; k++)
			{
			fSum += (A[(i*col) + k] * B[(k*col2) + j]);
			}
		count++;
		C[(i*col2) + j] = fSum;
		}
	}
}
void TransposeOnCPU(float *matrix, float *matrixTranspose, int row, int col)
{

	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
		matrixTranspose[j*row + i] = matrix[i*col + j];
		}
	}
}

void svd_2_3_alt(float *a, float *u,float *sig, float *vt, int m,int n)
{
	float T = 0;
	float D = 0;
	float lam1 = 0;
	float lam2 = 0;
	float u1_norm = 0;
	float u2_norm = 0;
	float sig_inv[m*m];
	float temp[m*m];
	float u_t[m*m];
	float aat[m*m];
	float temp_var;
	float fSum;
		
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

	print_matrix("final u ",m,m,u);
	print_matrix("final sigma ",m,m,sig);
	print_matrix("final vt ",m,n,vt);
}
/*
void svd_2_3_alt(float *a, float *u,float *sig, float *vt, int m,int n)
{
	float T = 0;
	float D = 0;
	float lam1 = 0;
	float lam2 = 0;
	float u1_norm = 0;
	float u2_norm = 0;
	float sig_inv[m*m];
	float temp[m*m];
	float aat[m*m];
	float temp_var;
	float fSum;
		
	aat[0] = a[0]*a[0]+ a[1]*a[1] + a[2]*a[2];
	aat[1] = a[0]*a[3]+ a[1]*a[4] + a[2]*a[5];
	aat[2] = a[3]*a[0]+ a[4]*a[1] + a[5]*a[2];
	aat[3] = a[3]*a[3]+ a[4]*a[4] + a[5]*a[5]; 

	print_matrix("aat ",m,m,aat);
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
	temp_var = u[1];
	u[1] = u[2];
	u[2] = temp_var;

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
			fSum += (sig_inv[(i*m) + k] * u[(k*m) + j]);
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
	print_matrix("final vt ",m,n,vt);
}
*/
void cpuTransMatrixMult(float *A, float *B, float *C, int row, int col)
{
        float fSum;
	for (int i = 0; i < row; i++)
	{		      
		for (int j = 0; j < row; j++)
		{
		fSum = 0.0;
			for (int k = 0; k < col; k++)
			{
			fSum += (A[(i*col) + k] * B[(k*row) + j]);
			}
		C[(i*row) + j] = fSum;
		}
	}
}


int main()
{
const int m = 2;
const int n = 3;
float a[m*n] = {1,2,3,2,2,1};
//float *at = new float [n*m];
float at[n*m] = {1,2,2,2,3,1};
float aat[m*m] = {0,0,0,0};
float u[m*m] = {0,0,0,0};
float sig[m*m] = {0,0,0,0};
float vt[m*n] = {0,0,0,0,0,0};

cpuTransMatrixMult(a, at, aat, m, n);
//print_matrix("aat",m,m,aat);

svd_2_3_alt(a,u,sig,vt,m,n);

}
