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

int main(void)
{
	const int row = 2;
	const int col = 15;
	const int row1 = 384;
	const int col1 = 15;


	float *xy = new float [row*col];
	float *mean = new float [row];
	float *B = new float [row1*col1];
	float *B_mean = new float [row1];
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
	cout << "value of data_size : "<< data_size << endl;	
	float *M = new float [row*row1];
	float *C = new float [data_size];
	float *E = new float [row*col];
	float *T = new float[row];

	initializeZero(M, row1,row);
	initializeZero(C, data_size, 1);
	initializeZero(E,col,row);
	rowMean(xy,col,row,T);

	delete[] xy;
        delete[] mean;
	delete[] B;
	delete[] B_mean;
	delete[] M;
	delete[] C;
	delete[] E;
	delete[] T;

}

