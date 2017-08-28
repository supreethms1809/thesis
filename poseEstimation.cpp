#include <stdio.h>
#include <string>
#include <iostream>
#include <fstream>
#include <math.h>
#include <stdlib.h>

int readValues(std::string text,float *variable,int i);

using std::string;
using namespace std;

int readValues(std::string text, float *variable, int i)
{
 	float temp;
	//cout << "value of text " << text << endl;
	ifstream myReadFile;
	myReadFile.open(text, ios::in);
	if (myReadFile.is_open()) 
	{
		while (!myReadFile.eof())
		{
			myReadFile >> temp;
			//cout << "value of variable " << variable[i] << endl;
			variable[i] = temp;
			i++;                                                                                                                                                                     }
	}
  	myReadFile.close();
	return i;
}

void displayValues(float *variable, int items)
{
	for (int i =0; i < items; i++)
	{
		cout << "Value of variable :"<< variable[i] << endl;
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

void calculateZ(float *Z,float *BBt,float *xy, float *E, float *T, float *B_transpose, float mu, float *M, float *Y,const int row,const int col,const int row1)
{
	float *temp = new float [row*col];
	float *temp2 = new float [row*row1];
	float *temp3 = new float [row*row1]; 
	float *Znum = new float [row*row1];
	float *Zden = new float [row1*row1];
	float *Zdenaug = new float [row1*row1*row1*row1];
	float *ZdenInverse = new float [row1*row1];


	delete [] temp;	
	delete [] temp2;
	delete [] temp3;
	delete [] Znum;
	delete [] Zden;
	delete [] Zdenaug;
	delete [] ZdenInverse;

}


int main(void)
{
	const int row = 2;
	const int col = 15;
	const int row1 = 384;
	const int col1 = 15;


	float *xy = new float [row*col];
	float *mean = new float [row];
	float *B = new float [row1*col1];
	float *B_transpose = new float [col1*row1];
	float *B_mean = new float [row1];
	float *BBt = new float [row1*row1];
	int items = 0;
	float a = 0.0f;
	int B_items = 0;

	items = readValues("exp.txt",xy,items);
	rowMean(xy, col, row, mean);
        Scalc(xy, col, row, mean);
        rowMean(xy, col, row, mean);
	a = mean_of_std_deviation(xy,col,row,mean);
	newScalc(xy,col,row,a);
	//displayValues(xy,items);
	B_items = readValues("exp1.txt", B, B_items);
	rowMean(B,col1,row1,B_mean);
	Scalc(B, col1,row1,B_mean);
	const int data_size = row1/3;	
	
	//ssr2D3D_alm
	//M => (2*384) = 0,  C ==> (1*384) = 0, E ==> (2*15) = 0, T ==> mean(W,2)
	//cout << "value of data_size : "<< data_size << endl;	
	float *M = new float [row*row1];
	float *C = new float [data_size];
	float *E = new float [row*col];
	float *T = new float[row];

	initializeZero(M, row1,row);
	initializeZero(C, data_size, 1);
	initializeZero(E,col,row);
	rowMean(xy,col,row,T);

	// auxiliary variables for ADMM
	float *Z = new float [row*row1];
	float *Y = new float [row*row1];
	float *Z0 = new float [row*row1];
	float *Q = new float [row*row1];
	float mu = 0.0f;
	initializeZero(Z,row1,row);
	initializeZero(Y,row1,row);

	mu = meanCalc(xy,col,row);
	//cout << "value of mu is " << mu << endl;

	TransposeOnCPU(B,B_transpose,row1,col);
	cpuTransMatrixMult(B, B_transpose, BBt, row1, col);
	initialize(Z0,Z,row1,row);


	delete[] xy;
        delete[] mean;
	delete[] B;
	delete[] B_transpose;
	delete[] B_mean;
	delete[] BBt;
	delete[] M;
	delete[] C;
	delete[] E;
	delete[] T;
	delete[] Z;
	delete[] Y;
	delete[] Z0;
	delete[] Q;

}

