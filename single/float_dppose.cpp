#include <stdio.h>
#include <string>
#include <iostream>
#include <fstream>
#include <math.h>
#include <stdlib.h>

using std::string;
using namespace std;

//extern void gpuInverseOfMatrix(double *h_matrix,double *h_iden_mat, int col);

int readValues(char *text, float *variable, int i, int row, int col)
{
 	float temp;
	cout << "value of text " << text << endl;
	ifstream myReadFile;
	myReadFile.open(text, ios::in);
	if (myReadFile.is_open()) 
	{
		while (!myReadFile.eof())
		{
			if(i < (row*col))
			{
				myReadFile >> temp;
				cout << "i = " << i<< endl;
				cout << "value of variable " << temp<< endl;
				variable[i] = temp;
				i++;     
			}
			else
			{
				break;
			}
		}
	}
  	myReadFile.close();
	return i;
}

int main(void)
{

	const int row = 2;
	const int col = 15;
	const int row1 = 384;
	const int col1 = 15;
	float tol = 1e-04;

	float *xy = new float [row*col];
	int items = 0;

	items = readValues("messi2.txt",xy,items,row,col);
	cout << "items = "<<items<<endl;

	delete[] xy;
}

