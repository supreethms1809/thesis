#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <math.h>
#include <mkl.h>
#include <limits>
#include <chrono>
#include <ctime>
#include <string>

using namespace std;

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


void cpuInverseOfMatrix(double *matrix, int col)
{
	for (int m = 0; m < col; m++)
	{
		//Checking if diagonal element is 0
		if (matrix[((2 * col) + 1)*m] == 0)
		{
			//checking if the row is last row. If it is last row add the previous row to make it non zero
                	if (m == (col - 1))
			{
				for (int i = 0; i < (2 * col); i++)
				{					
				matrix[(m * (2 * col)) + i] = matrix[((m - 1) * (2 * col)) + i] + matrix[(m * (2 * col)) + i];
				}
			}
			else	//if it is not last row, add the next row.
			{
			        for (int i = 0; i < (2 * col); i++)
				{
				matrix[(m * 2 * col) + i] = matrix[((m + 1) * 2 * col) + i] + matrix[(m * 2 * col) + i];
				}
			}
		}
		//Make the diagonal elements 1 along with the whole row(divide).
		double initialValue = matrix[((2 * col) + 1)*m];
		for (int j = 0; j < (2 * col); j++)
		{
		matrix[(m * (2 * col)) + j] = matrix[(m * (2 * col)) + j] / initialValue;
		}

		//Making the elements of the row to zero
		for (int k = 0; k < col; k++)
		{
			float tempIni;
			tempIni = matrix[m + (k * (2 * col))];
			if (k == m)
			{
			//Just a loop to do nothing
			}
			else
			{
				for (int l = 0; l < (2 * col); l++)
				{
				
				double tempMul, tempDiv;
				tempMul = matrix[(m * (2 * col)) + l] * tempIni;
				tempDiv = tempMul / matrix[(m * (2 * col)) + m];
				matrix[(k * 2 * col) + l] = matrix[(k * (2 * col)) + l] - tempDiv;
				}
                        }

                }
        }
}

int main()
{
	const int n = 3;
	double a[n*n] = {0,3,4,1,3,10,4,9,16};

	print_matrix("Input",a,n,n);
    	cout << endl << endl;

    	cpuInverseOfMatrix(a,n);

	print_matrix("Inverse - CPU",a,n,n);

	return 0;
}
