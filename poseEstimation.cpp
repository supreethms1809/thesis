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
		

int main(void)
{
	const int row = 2;
	const int col = 15;
	const int row1 = 384;
	const int col1 = 15;


	float *xy = new float [row*col];
	float *mean = new float [row];
	int items = 0;

	items = readValues("exp.txt",xy,items);
	rowMean(xy, col, row, mean);
        Scalc(xy, col, row, mean);
        rowMean(xy, col, row, mean);


	delete[] xy;
        delete[] mean;


}

