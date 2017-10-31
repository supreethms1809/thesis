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
    	for(int i=0; i<n; i++)
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

int main(void)
{
	const int n = 3;
	high_resolution_clock::time_point t3,t4;
/*	const int n = 384;
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
*/
	float a[n*n] = {0,3,4,1,3,10,4,9,16};
/*	float a[n*n] = {
        0.378589,   0.971711,   0.016087,   0.037668,   0.312398,
        0.756377,   0.345708,   0.922947,   0.846671,   0.856103,
        0.732510,   0.108942,   0.476969,   0.398254,   0.507045,
        0.162608,   0.227770,   0.533074,   0.807075,   0.180335,
        0.517006,   0.315992,   0.914848,   0.460825,   0.731980
    };
*/
	float *I = new float [n*n];

	eye(I,n,n);
/*	print_matrix("Input",a,n,n);
	print_matrix("Identity matrix",I,n,n);
    	cout << endl << endl;
*/	

	t3 = high_resolution_clock::now();
    	cpuInverseOfMatrix(a,I,n);

	t4 = high_resolution_clock::now();
	duration<float> time_span1 = duration_cast<duration<float>>(t4 - t3);
	cout << "Time in miliseconds is : " << time_span1.count() * 1000 << " ms" << endl;
	

	cout << "CPU inverse "<<endl<<endl;
	print_matrix("Inverse - Identity",a,n,n);
	print_matrix("Inverse ",I,n,n);

	float h_a[n*n] = {0,3,4,1,3,10,4,9,16};
/*	float h_a[n*n] = {
        0.378589,   0.971711,   0.016087,   0.037668,   0.312398,
        0.756377,   0.345708,   0.922947,   0.846671,   0.856103,
        0.732510,   0.108942,   0.476969,   0.398254,   0.507045,
        0.162608,   0.227770,   0.533074,   0.807075,   0.180335,
        0.517006,   0.315992,   0.914848,   0.460825,   0.731980
    };
*/
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

}
