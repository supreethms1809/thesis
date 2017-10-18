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

void print_matrix(string desc,double *a, int m, int n)
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

void eye(double *I, int m, int n)
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

void cpuInverseOfMatrix(double *matrix,double *iden_mat, int col)
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
		//Make the diagonal elements 1 along with the whole row(divide).
		double initialValue = matrix[((col) + 1)*m];
		cout << "value of initial value is :" << initialValue << endl;
		for (int j = 0; j < (col); j++)
		{
		matrix[(m * (col)) + j] = matrix[(m * (col)) + j] / initialValue;
		iden_mat[(m * (col)) + j] = iden_mat[(m * (col)) + j] / initialValue;
		}
		print_matrix("Intemediate",matrix,col,col);
		print_matrix("Inter identity",iden_mat,col,col);
		//Making the elements of the row to zero
		for (int k = 0; k < col; k++)
		{
			double tempIni,tempIni1;
			tempIni = matrix[m + (k * (col))];
			cout << "k = "<<k<<"\t tempIni =  "<<tempIni <<endl;
			if (k == m)
			{
			//Just a loop to do nothing
			}
			else
			{
				for (int l = 0; l < (col); l++)
				{
				
				double tempMul, tempDiv, tempMul1, tempDiv1;
				cout << "m = "<<m<<" l = "<<l<<" \t value of m*col+l "<< (m * ( col)) + l << "  \t matrix value "<< matrix[(m * ( col)) + l] << endl;
				tempMul = matrix[(m * ( col)) + l] * tempIni;
				tempMul1 = iden_mat[(m*col)+l] * tempIni;
				
				cout <<	"tempMul"<<tempMul<<endl;
			
				tempDiv = tempMul / matrix[(m * (col)) + m];
				tempDiv1 = tempMul1 / matrix[(m * (col)) + m];
				
				cout <<"tempDiv" << matrix[(m * (col)) + m] <<endl;				
				cout <<"tempDiv - Identity" << iden_mat[(m * (col)) + m] <<endl;				

				matrix[(k * col) + l] = matrix[(k * (col)) + l] - tempDiv;
				cout << "value inside :" <<matrix[(k*col)+l]<<endl;
				
				iden_mat[(k*col)+l] = iden_mat[(k*col)+l] - tempDiv1;
				cout << "value of identity  :" << iden_mat[(k*col)+l]<<endl;
				}
                        }

                }
        }
}

int main(void)
{
	const int n = 3;
	double a[n*n] = {0,3,4,1,3,10,4,9,16};
	double *I = new double [n*n];

	eye(I,n,n);
	print_matrix("Input",a,n,n);
	print_matrix("Identity matrix",I,n,n);
    	cout << endl << endl;
	

    	cpuInverseOfMatrix(a,I,n);

	print_matrix("Inverse - Identity",a,n,n);
	print_matrix("Inverse ",I,n,n);

}
