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
#include <string>

using namespace std;
using namespace std::chrono;
using std::string;

extern void gpuInverseOfMatrix(float *h_matrix,float *h_iden_mat, int col);

void print_matrix(string desc,float *a, int m, int n)
{
	cout << desc << endl;
    	for(int i=0; i<m; i++)
	{
        	for(int j=0; j<n; j++)
		{
		cout << a[i*n+j] << "\t";
		}
        cout << endl;
    	}
}

void eye(float *I, int m, int n)
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

void cpuInverseOfMatrix(float *matrix,float *iden_mat, int col)
{
	for (int m = 0; m < col; m++)
	{
		//Checking if diagonal element is 0
		if (matrix[((col) + 1)*m] == 0)
		{
			//checking if the row is last row. If it is last row add the previous row to make it non zero
                	if (m == (col - 1))
			{
				for (int i = 0; i < (col); i++)
				{					
				matrix[(m * (col)) + i] = matrix[((m - 1) * (col)) + i] + matrix[(m * (col)) + i];
				iden_mat[(m * (col)) + i] = iden_mat[((m - 1) * (col)) + i] + iden_mat[(m * (col)) + i];
				}
			}
			else	//if it is not last row, add the next row.
			{
			        for (int i = 0; i < (col); i++)
				{
				matrix[(m * col) + i] = matrix[((m + 1) * col) + i] + matrix[(m * col) + i];
				iden_mat[(m * col) + i] = iden_mat[((m + 1) * col) + i] + iden_mat[(m * col) + i];
				}
			}
		}
	}
	for(int m=0;m<col;m++)
	{
		//Make the diagonal elements 1 along with the whole row(divide).
		float initialValue = matrix[((col) + 1)*m];
		
		for (int j = 0; j < (col); j++)
		{
		matrix[(m * (col)) + j] = matrix[(m * (col)) + j] / initialValue;
		iden_mat[(m * (col)) + j] = iden_mat[(m * (col)) + j] / initialValue;
		}

		float tempDen;
		tempDen = matrix[(m * (col)) + m];
	
		//Making the elements of the row to zero
		for (int k = 0; k < col; k++)
		{	
			float tempIni,tempIni1;
			tempIni = matrix[m + (k * (col))]/tempDen;
			if (k != m)
			{
				for (int l = 0; l < (col); l++)
				{
				matrix[(k * col) + l] = matrix[(k * (col)) + l] - (matrix[(m * ( col)) + l] * tempIni);
				iden_mat[(k*col)+l] = iden_mat[(k*col)+l] - (iden_mat[(m*col)+l] * tempIni);
				}
                        }

                }
        }
}

void inv_mu_i(float mu,float *I,int m)
{
	for(int i=0;i<m;i++)
	{
		for(int j=0;j<m;j++)
		{
			if(i == j)
			{
			I[(i*m)+j] = 1 / (I[(i*m)+j] * mu);
			}
		}
	}
}
void cpuMatrixMult(float *A, float *B, float *C, int row, int col,int col2)
{
        float fSum;
//        int count = 0;
        for (int i = 0; i < row; i++)
        {
                for (int j = 0; j < col2; j++)
                {
                fSum = 0.0;
                        for (int k = 0; k < col; k++)
                        {
                        fSum += (A[(i*col) + k] * B[(k*col2) + j]);
                        }
//                count++;
                C[(i*col2) + j] = fSum;
                }
        }
}

lapack_int matInv(float *A, int n)
{
	int ipiv[n+1];
	lapack_int ret;
	
	ret = LAPACKE_sgetrf(LAPACK_ROW_MAJOR,n,n,A,n,ipiv);
	
	if(ret != 0)
	{
		return ret;
	}
	
	ret = LAPACKE_sgetri(LAPACK_ROW_MAJOR,n,A,n,ipiv);
	return ret;

}

void zero(float *A, int n)
{
	for(int i=0;i<n;i++)
	{
		for(int j=0;j<n;j++)
		{
			A[(i*n)+j] = 0.0f;
		}
	}

}

void add(float *temp1,int n)
{
	for(int i=0;i<n;i++)
	{
		for(int j=0;j<n;j++)
		{
			if(i==j)
			{
			temp1[(i*n)+j] = temp1[(i*n)+j] + 1.0f;
			}
		}
	}

}

void sub(float *I_n,float *temp1,float *inv,int m,int n)
{
	for(int i=0;i<n;i++)
	{
		for(int j=0;j<n;j++)
		{
			inv[(i*n)+j] = I_n[(i*n)+j] - temp1[(i*n)+j];
		}
	}


}

/*
void Zden_cacl_old(float *B, float * B_transpose, float *Zden,float mu,int m,int n)
{
        float *I_m = new float [m*m];
	float *temp = new float [m*n];
	float *temp1 = new float [n*n];
	float *temp2 = new float [n*m];
	float *temp3 = new float [n*m];
	float *temp4 = new float [m*m];
	float *temp5 = new float [m*m];
        eye(I_n,2,2);
        eye(I_m,2,2);

	inv_mu_i(mu,I_m,m);
	cpuMatrixMult(I_m,B,temp,m,m,n);
	cpuMatrixMult(B_transpose,temp,temp1,n,m,n);
	add(temp1,n);
	matInv(temp1,n);	
	cpuMatrixMult(temp1,B_transpose,temp2,n,n,m);
	cpuMatrixMult(temp2,I_m,temp3,n,m,m);
	cpuMatrixMult(B,temp3,temp4,m,n,m);
	cpuMatrixMult(I_m,temp4,temp5,m,m,m);
	sub(I_m,temp5,Zden,m,m);

	delete[] I_m;
	delete[] temp;
	delete[] temp1;
	delete[] temp2;
	delete[] temp3;
	delete[] temp4;
	delete[] temp5;
}
*/

void Zden_cacl(float *B, float * B_transpose, float *Zden,float mu,const int m,const int n)
{
        float *I_m = new float [m*m];
	float *temp = new float [m*n];
	float *temp1 = new float [n*n];
	float *temp2 = new float [n*m];
	float *temp3 = new float [n*m];
	float *temp4 = new float [m*m];
	float *temp5 = new float [m*m];
        eye(I_m,m,m);
	cout << "coming inside Zden_calc"<<endl;

	inv_mu_i(mu,I_m,m);	
	
	cpuMatrixMult(I_m,B,temp,m,m,n);
	cout << "checkpoint 1"<<endl;
	cpuMatrixMult(B_transpose,temp,temp1,n,m,n);
	
	add(temp1,n);
	matInv(temp1,n);	
	cpuMatrixMult(temp1,B_transpose,temp2,n,n,m);
	cpuMatrixMult(temp2,I_m,temp3,n,m,m);
	cpuMatrixMult(B,temp3,temp4,m,n,m);
	cpuMatrixMult(I_m,temp4,temp5,m,m,m);
	sub(I_m,temp5,Zden,m,m);
	cout << "leaving Zden_calc"<<endl;

	delete[] I_m;
	delete[] temp;
	delete[] temp1;
	delete[] temp2;
	delete[] temp3;
	delete[] temp4;
	delete[] temp5;
}



int main(void)
{
/*	const int n = 3;
	const int m = 3;
	high_resolution_clock::time_point t3,t4;

	const int n = 384;
	float *a = new float[n*n];
	float *h_a = new float[n*n];

	high_resolution_clock::time_point t3,t4;
	for (int i = 1; i < n; i++) 
	{
		for (int j = 1; j < n; j++)
		{
			a[i*n+j]=rand()% 100 +1;
			h_a[i*n+j] = a[i*n+j];
		}
	}

	float a[n*n] = {0,3,4,1,3,10,4,9,16};
	float a[n*n] = {
        0.378589,   0.971711,   0.016087,   0.037668,   0.312398,
        0.756377,   0.345708,   0.922947,   0.846671,   0.856103,
        0.732510,   0.108942,   0.476969,   0.398254,   0.507045,
        0.162608,   0.227770,   0.533074,   0.807075,   0.180335,
        0.517006,   0.315992,   0.914848,   0.460825,   0.731980
    };

	float *I = new float [n*n];

	eye(I,n,n);
	print_matrix("Input",a,n,n);
	print_matrix("Identity matrix",I,n,n);
    	cout << endl << endl;
	

	t3 = high_resolution_clock::now();
    	cpuInverseOfMatrix(a,I,n);

	t4 = high_resolution_clock::now();
	duration<float> time_span1 = duration_cast<duration<float>>(t4 - t3);
	cout << "Time in miliseconds is : " << time_span1.count() * 1000 << " ms" << endl;
	

	cout << "CPU inverse "<<endl<<endl;
	print_matrix("Inverse - Identity",a,n,n);
	print_matrix("Inverse ",I,n,n);

	float h_a[n*n] = {0,3,4,1,3,10,4,9,16};
	float h_a[n*n] = {
        0.378589,   0.971711,   0.016087,   0.037668,   0.312398,
        0.756377,   0.345708,   0.922947,   0.846671,   0.856103,
        0.732510,   0.108942,   0.476969,   0.398254,   0.507045,
        0.162608,   0.227770,   0.533074,   0.807075,   0.180335,
        0.517006,   0.315992,   0.914848,   0.460825,   0.731980
    };

        float *h_I = new float [n*n];

        eye(h_I,n,n);
	t3 = high_resolution_clock::now();
	
	gpuInverseOfMatrix(h_a,h_I,n);
//    	cout << endl << endl;
	t4 = high_resolution_clock::now();
	duration<float> time_span = duration_cast<duration<float>>(t4 - t3);
	cout << "Time in miliseconds is : " << time_span.count() * 1000 << " ms" << endl;
	

	cout << "GPU inverse "<<endl<<endl;
	print_matrix("h_a ",h_a,n,n);
	print_matrix("h_I",h_I,n,n);

        float *I_n = new float [2*2];
        float *I_m = new float [2*2];
        eye(I_n,2,2);
        eye(I_m,2,2);
	float temp[2*2] = {0,0,0,0};
	float temp1[2*2] = {0,0,0,0};
	float *inv = new float[2*2];
	float B[2*2] = {1,2,2,1};
	float B_transpose[2*2] = {1,2,2,1};
//Y =  inv(mu*I)-inv(mu*I)*B*inv((inv(I1)+B'*inv(mu*I)*B))*B'*inv(mu*I);
//A=[3, 0; 0,3;];
//U=[1,4;2,3;];
//C=[1,0; 0,1;];
//V = [3 ,4; 5,2;];
//inv3 = inv(A)-inv(A)*U*inv((inv(C)+V*inv(A)*U))*V*inv(A)
// A = I 
// U = B
// V = B'
// C = I1
	float mu = 1.140937;
	inv_mu_i(mu,I_n,2);
	cpuMatrixMult(I_n,B,temp,2,2,2);
	cpuMatrixMult(B_transpose,temp,temp1,2,2,2);
	add(temp1,2);
	matInv(temp1,2);	
	zero(temp,2);
	cpuMatrixMult(temp1,B_transpose,temp,2,2,2);
	zero(temp1,2);
	cpuMatrixMult(temp,I_n,temp1,2,2,2);
	zero(temp,2);
	print_matrix("temp1",temp1,2,2);
	cpuMatrixMult(B,temp1,temp,2,2,2);
	
	zero(temp1,2);
	cpuMatrixMult(I_n,temp,temp1,2,2,2);
	
	sub(I_n,temp1,inv,2,2);
	print_matrix("inverse",inv,2,2);

	delete[] I_n;
	delete[] I_m;
*/
	//float *D = new float [5*3];
	//float *D_t = new float [3*5];
	float mu = 1.140937;
	float *B = new float [384*15];
	float *B_t = new float [15*384];
	float D[5*3] = {3,2,2,4,2,1,1,6,5,2,2,1,4,6,3};
	float D_t[3*5] = {3,4,1,2,4,2,2,6,2,6,2,1,5,1,3};
	print_matrix("D",D,5,3);
	print_matrix("D_t",D_t,3,5);
	float *inv = new float [5*5];

	Zden_cacl(D, D_t, inv,mu,5,3);
	print_matrix("inv",inv,5,5);
	delete[] B;
	delete[] B_t;
	delete[] inv;
}
