#include <stdio.h>
#include <string>
#include <iostream>
#include <fstream>
#include <math.h>
#include <stdlib.h>



int readValues(std::string text, float *variable, int i)
{
	float temp;
	//cout << "value of text " << text << endl;
	ifstream myReadFile;
	myReadFile.open(text, ios::in);
	if (myReadFile.is_open()) {
		while (!myReadFile.eof()) 
		{
			myReadFile >> temp;
			//cout << "value of variable " << variable[i] << endl;
			variable[i] = temp;
			i++;
		}
	}
	myReadFile.close();
	return i;
}


void displayValues(float *variable, int items)
{

	for (int i = 0;i < items;i++)
	{
		//if(i<30)
		//{
		//cout << "value of i " << i << endl;
			cout << "value of variable " << variable[i] << endl;
		//}
	}

}

void rowMean(float *variable, int col, int row , float *mean)
{
	float sum;

	for (int j = 0;j < row; j++)
	{
		sum = 0;
		for (int i = 0;i < col; i++)
		{
			sum += variable[(j*col)+ i];
		}
		mean[j] = sum / col;
		//cout << "value of mean " << mean[j] << endl;
	}
}


void Scalc(float *variable, int col, int row, float *mean)
{
	for (int j = 0;j < row;j++)
	{
		for (int i = 0;i < col;i++)
		{
			
			variable[(j * col) + i] = variable[(j * col) + i] - mean[j];
		}
	}
}

float mean_of_std_deviation(float *variable, int col, int row, float *mean)
{
	float std[2];
	float temp,a;
	for (int j = 0;j < row;j++)
	{
		temp = 0;
		for (int i = 0;i < col;i++)
		{
			temp += pow((variable[(j * 15) + i] - mean[j]), 2);
		}
		std[j] = sqrt(temp / (col));
		//cout << "value od std deviation " << std[j] << endl;
	}
	a = (std[0] + std[1]) / 2;
	//cout << "value of a " << a << endl;
	return a;
}


void newScalc(float *variable, int col, int row, float a)
{
	for (int i = 0;i < row;i++)
	{
		for (int j = 0;j < col;j++)
		{
			variable[(i * col) + j] = variable[(i * col) + j] / a;

		}
	}

}

void initializeZero(float *variable, int col, int row)
{
	for (int i = 0;i < row;i++)
	{
		for (int j = 0;j < col;j++)
		{
			variable[(i * col) + j] = 0.0f;

		}
	}
}

float meanCalc(float *variable, int col, int row)
{
	float sum = 0;
	float mean = 0;
	float mu = 0;

	for (int i = 0;i < row; i++)
	{
		for (int j = 0;j < col; j++)
		{
			sum += abs(variable[(i*col) + j]);
			//cout << "value of varialbe is " << variable[(j*col) + i] << endl;
		}
	}
	mean = sum / (col*row);
	//cout << "value of mean " << mean << endl;
	mu = 1 / mean;
	return mu;
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



void initialize(float *variable,float *variable2, int col, int row)
{
	for (int i = 0;i < row;i++)
	{
		for (int j = 0;j < col;j++)
		{
			variable[(i * col) + j] = variable2[(i * col) + j];

		}
	}
}

void cpuTransMatrixMult(float *A, float *B, float *C, int row, int col)
{
	float fSum;


	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < row; j++)
		{
			fSum = 0.0f;
			for (int k = 0; k < col; k++)
			{
				fSum += (A[(i*col) + k] * B[(k*row) + j]);
			}

			C[(i*row) + j] = fSum;
		}
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
			fSum = 0.0f;
			for (int k = 0; k < col; k++)
			{
				fSum += (A[(i*col) + k] * B[(k*col2) + j]);
			}
			//cout << count<<"\t" <<"value of (i*row) + j \t"<< (i*row) + j <<"\t value of i and j"<<"\t"<<i<<" and "<<j<<"\t value of fsum" << fSum << endl;
			count++;
			C[(i*col2) + j] = fSum;
		}
	}
	//cout << "count =" << count << endl;
}

void scalarToMatrixMultiply(float *Temp, float *M, float mu, int row, int col)
{
	for (int i = 0;i < row;i++)
	{
		for (int j = 0;j < col;j++)
		{
			Temp[(i*col) + j] = mu * M[(i*col) + j];
		}
	}

}

void sumOfMatrix(float *Znum,float *temp2, float *temp3, float *temp4, int row, int col)
{
	for (int i = 0;i < row;i++)
	{
		for (int j = 0;j < col;j++)
		{
			Znum[(i*col) + j] = temp2[(i*col) + j] + temp3[(i*col) + j] + temp4[(i*col) + j];
		}
	}
}

void diferenceOfMatrix(float *diffMatrix, float *matrix1, float *matrix2, int row, int col)
{
	for (int i = 0;i < row;i++)
	{
		for (int j = 0;j < col;j++)
		{
			diffMatrix[(i*col) + j] = matrix1[(i*col) + j] - matrix2[(i*col) + j] ;
		}
	}
}

void AugmentIdentity(float *matrix, float *augmatrix, int n)
{
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			augmatrix[(i * 2 * n) + j] = matrix[(i*n) + j];
			augmatrix[(((2 * i) + 1)*n) + j] = 0.0f;
		}
	}
	for (int i = 0; i < n; i++)
	{
		augmatrix[(((2 * i) + 1)*n) + i] = 1.0f;

	}
}


void cpuInverseOfMatrix(float *matrix, int n)
{
	for (int m = 0; m < n; m++)
	{
		//Checking if diagonal element is 0
		if (matrix[((2 * n) + 1)*m] == 0)
		{
			if (m == (n - 1))
			{
				for (int i = 0; i < (2 * n); i++)
				{
					matrix[(m * 2 * n) + i] = matrix[((m - 1) * 2 * n) + i] + matrix[(m * 2 * n) + i];
				}
			}
			else
			{
				for (int i = 0; i < (2 * n); i++)
				{
					matrix[(m * 2 * n) + i] = matrix[((m + 1) * 2 * n) + i] + matrix[(m * 2 * n) + i];
				}
			}
		}

		//Make the diagonal elements 1 along with the whole row(divide).
		float initialValue = matrix[((2 * n) + 1)*m];
		for (int j = 0; j < (2 * n); j++)
		{

			matrix[(m * 2 * n) + j] = matrix[(m * 2 * n) + j] / initialValue;

		}

		//Making the elements of the row to zero
		for (int k = 0; k < n; k++)
		{
			float tempIni;
			tempIni = matrix[m + (k * 2 * n)];
			if (k == m)
			{
				//Just a loop to do nothing
			}
			else
			{
				for (int l = 0; l < (2 * n); l++)
				{
					float tempMul, tempDiv;
					tempMul = matrix[(2 * m*n) + l] * tempIni;
					tempDiv = tempMul / matrix[(2 * m*n) + m];

					matrix[(k * 2 * n) + l] = matrix[(k * 2 * n) + l] - tempDiv;

				}
			}
		}
	}


}



void addScalarToDiagonal(float *Zden, float *BBt, float mu, int row, int col)
{
	for (int i = 0;i < row;i++)
	{
		for (int j = 0;j < col;j++)
		{
			if(i==j)
			{
				Zden[(i*col) + j] = BBt[(i*col) + j] + mu;
			}
			else
			{
				Zden[(i*col) + j] = BBt[(i*col) + j] + 0.0f;
			}
		}
	}
}

void Inverse(float *augmatrix, float *matrixInverse, int n)
{
	for (int i = 0;i < n;i++)
	{
		for (int j = 0;j < n;j++)
		{
			matrixInverse[(i*n) + j] = augmatrix[(i*2*n)+n+j];
		}
	}

}

void calculateZ(float *Z,float *BBt,float *xy, float *E, float *T, float *B_transpose, float mu, float *M, float *Y,const int row,const int col,const int row1)
{
	
	float *temp = (float*)_aligned_malloc(row*col*sizeof(float), 16);
	float *temp2 = (float*)_aligned_malloc(row*row1*sizeof(float), 16);
	float *temp3 = (float*)_aligned_malloc(row*row1*sizeof(float), 16);
	float *Znum = (float*)_aligned_malloc(row*row1*sizeof(float), 16);
	float *Zden = (float*)_aligned_malloc(row1*row1*sizeof(float), 16);
	float *Zdenaug = (float*)_aligned_malloc(row1*row1*row1*row1*sizeof(float), 16);
	float *ZdenInverse = (float*)_aligned_malloc(row1*row1*sizeof(float), 16);
	

	//numerator
	//temp = (W-E-T*ones(1,p))
	for (int i = 0;i < row;i++)
	{
		for (int j = 0;j < col;j++)
		{
			temp[(i*col) + j] = xy[(i*col) + j] - E[(i*col) + j] - T[i];
		}
	}
	//displayValues(temp, row*col);  
	
	//temp2 = temp * B'
	cpuMatrixMult(temp, B_transpose, temp2, row, col, row1);
	//displayValues(temp2,row*row1);

	//temp3 = mu*M
	scalarToMatrixMultiply(temp3, M, mu, row, row1);
	//displayValues(temp3, row*row1);

	//Znum = ((W-E-T*ones(1,p))*B'+mu*M+Y)
	sumOfMatrix(Znum,temp2, temp3, Y, row, row1);
	//displayValues(Znum, row*row1);
	//cout << "value of mu" << mu << endl;

	//denominator
	addScalarToDiagonal(Zden,BBt,mu,row1,row1);
	//displayValues(Zden, row1*row1);

	//Inverse calculation via guass-jordon method
	AugmentIdentity(Zden, Zdenaug, row1);
	cpuInverseOfMatrix(Zdenaug, row1);
	//displayValues(Zdenaug, row1*row1);
	Inverse(Zdenaug,ZdenInverse,row1);
	//displayValues(ZdenInverse, row1*row1);

	//Z = ((W-E-T*ones(1,p))*B'+mu*M+Y)/(BBt+mu*eye(3*k))
	cpuMatrixMult(Znum, ZdenInverse, Z, row, row1, row1);
	//displayValues(Z, row*row1);


	_aligned_free(temp);
	_aligned_free(temp2);
	_aligned_free(temp3);
	_aligned_free(Znum);
	_aligned_free(Zden);
	_aligned_free(Zdenaug);
	_aligned_free(ZdenInverse);
	
}




void calculateQ(float *Q, float *Z, float *Y, float mu, int row, int row1)
{
	float *temp = (float*)_aligned_malloc(row*row1*sizeof(float), 16);
	scalarToMatrixMultiply(temp, Y, 1 / mu, row, row1);
	diferenceOfMatrix(Q, Z, temp, row, row1);
	displayValues(Q,row*row1);
	_aligned_free(temp);
}

//void prox_2norm(float *Q, float *M, float *C, float constant, int row, int col, int data_size)
//{
//	float *Qtemp = (float*)_aligned_malloc(int(6)*sizeof(float), 16);
//	float *work = (float*)_aligned_malloc(int(6)*sizeof(float), 16);
//	float *U = (float*)_aligned_malloc(int(4)*sizeof(float), 16);
//	float *V = (float*)_aligned_malloc(int(6)*sizeof(float), 16);
//	float *sigma = (float*)_aligned_malloc(int(9)*sizeof(float), 16);
//
//	int Qtemprow = 2;
//	int Qtempcol = 3;
//	int info = 0;
//	
//
//	for (int i = 0;i < data_size;i++)
//	{
//		for (int j = 0;j < 2;j++)
//		{
//			for (int k = 0;k < 3;k++)
//			{
//				Qtemp[(j * 3) + k] = Q[(3 * i) + (j*col) + k];
//				
//				//dgesvd("ALL", "ALL", &Qtemprow, &Qtempcol, Qtemp, &Qtemprow, sigma, U, &Qtemprow, V, &Qtempcol, work, lwork, &info);
//			}
//		}
//		LAPACKE_sgesvd(LAPACK_ROW_MAJOR, 'A', 'A', Qtemprow, Qtempcol, Qtemp, Qtemprow, sigma, U, Qtemprow, V, Qtempcol, work);
//	}
//
//	_aligned_free(Qtemp);
//	_aligned_free(work);
//	_aligned_free(U);
//	_aligned_free(V);
//	_aligned_free(sigma);
//}

